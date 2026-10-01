"""Loopback-only local control API and transcript event stream."""

from __future__ import annotations

import asyncio
import hmac
import json
import secrets
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from meetagent.ui.service import MeetingService

WEB_ROOT = Path(__file__).with_name("web")


class StartMeetingRequest(BaseModel):
    output_dir: str | None = None
    language: str = Field(default="auto", pattern="^(auto|zh|en|de)$")
    summary_language: str = Field(default="zh", pattern="^(zh|en|de)$")
    live: bool = False
    consent_to_stream: bool = False
    consent_to_summary_upload: bool = False
    # voice: speak answers; text: print only, no Gemini Live. None = config default.
    reply: str | None = Field(default=None, pattern="^(voice|text)$")


def create_app(service: MeetingService | None = None, token: str | None = None) -> FastAPI:
    service = service or MeetingService()
    token = token or secrets.token_urlsafe(32)
    app = FastAPI(title="MeetAgent Local GUI", docs_url=None, redoc_url=None)
    app.state.meeting_service = service
    app.state.local_token = token

    @app.middleware("http")
    async def restrict_to_loopback(request: Request, call_next):
        host = (request.headers.get("host") or "").split(":", 1)[0].strip("[]").lower()
        if host not in {"localhost", "127.0.0.1", "::1"}:
            return JSONResponse({"detail": "仅允许本机访问。"}, status_code=403)
        origin = request.headers.get("origin")
        if origin:
            parsed = urlsplit(origin)
            origin_host = (parsed.hostname or "").lower()
            request_host = host
            if parsed.scheme != "http" or origin_host not in {"localhost", "127.0.0.1", "::1"} or origin_host != request_host:
                return JSONResponse({"detail": "拒绝跨站请求。"}, status_code=403)
        if request.url.path.startswith("/api/") and request.url.path != "/api/bootstrap":
            supplied = request.headers.get("x-meetagent-token", "")
            if not hmac.compare_digest(supplied, token):
                return JSONResponse({"detail": "本地会话令牌无效。请刷新页面。"}, status_code=401)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "default-src 'self'; connect-src 'self' ws://localhost:* ws://127.0.0.1:*; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; object-src 'none'; base-uri 'none'"
        return response

    @app.get("/")
    async def home():
        return FileResponse(WEB_ROOT / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/api/bootstrap")
    async def bootstrap():
        return {"token": token}

    @app.get("/api/status")
    async def status():
        return service.status()

    @app.post("/api/meetings", status_code=202)
    async def start_meeting(payload: StartMeetingRequest):
        try:
            run = await service.start(
                output_dir=Path(payload.output_dir).expanduser() if payload.output_dir else None,
                language=payload.language, summary_language=payload.summary_language,
                live=payload.live, consent_to_stream=payload.consent_to_stream,
                consent_to_summary_upload=payload.consent_to_summary_upload,
                reply=payload.reply,
            )
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"meeting_id": run.meeting_id, "state": "starting", "transcript": str(run.transcript_path)}

    @app.post("/api/meetings/{meeting_id}/stop", status_code=202)
    async def stop_meeting(meeting_id: str):
        try:
            await service.stop(meeting_id)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"meeting_id": meeting_id, "state": "stopped"}

    @app.get("/api/meetings/{meeting_id}/transcript")
    async def transcript(meeting_id: str, offset: int = 0, limit: int = 500):
        try:
            entries = service.transcript(meeting_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        limit = min(max(limit, 1), 1000)
        return {"entries": entries[max(offset, 0):max(offset, 0) + limit], "total": len(entries)}

    @app.patch("/api/meetings/{meeting_id}/speaker")
    async def correct_speaker(meeting_id: str, payload: dict[str, Any]):
        try:
            result = service.correct_speaker(
                meeting_id, float(payload["timestamp"]), str(payload["name"]),
            )
        except (KeyError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc
        return result

    @app.post("/api/meetings/{meeting_id}/speaker/undo")
    async def undo_speaker(meeting_id: str):
        try:
            return service.undo_speaker_correction(meeting_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/meetings/{meeting_id}/files/{kind}")
    async def meeting_file(meeting_id: str, kind: str):
        try:
            run = service._find_meeting(meeting_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        suffixes = {"audio": ".wav", "transcript": ".txt", "jsonl": ".jsonl", "summary": ".md"}
        if kind not in suffixes:
            raise HTTPException(404, "未知文件类型。")
        path = run.transcript_path.with_suffix(suffixes[kind])
        if not path.is_file():
            raise HTTPException(404, "文件尚未生成。")
        return FileResponse(path, filename=path.name)

    @app.websocket("/api/events")
    async def events(websocket: WebSocket):
        host = (websocket.headers.get("host") or "").split(":", 1)[0].strip("[]").lower()
        origin = websocket.headers.get("origin")
        parsed = urlsplit(origin) if origin else None
        if host not in {"localhost", "127.0.0.1", "::1"} or (
            parsed and (parsed.scheme != "http" or parsed.hostname != host)
        ):
            await websocket.close(code=1008, reason="loopback origin required")
            return
        if not hmac.compare_digest(websocket.query_params.get("token", ""), token):
            await websocket.close(code=1008, reason="invalid local token")
            return
        await websocket.accept()
        run = service.last_meeting
        entries: list[dict[str, Any]] = []
        last_status: str | None = None
        sequence = 0
        summary_sent_for: str | None = None
        try:
            await websocket.send_json({"type": "status", "sequence": sequence, "data": service.status()})
            while True:
                current_status = service.status()
                encoded_status = json.dumps(current_status, sort_keys=True, ensure_ascii=False)
                if encoded_status != last_status:
                    sequence += 1
                    await websocket.send_json({"type": "status", "sequence": sequence, "data": current_status})
                    last_status = encoded_status
                current_run = service.last_meeting
                if current_run and (run is None or current_run.meeting_id != run.meeting_id):
                    run = current_run
                    entries = []
                if run:
                    fresh = service.transcript(run.meeting_id)
                    if fresh != entries:
                        is_append = len(fresh) >= len(entries) and fresh[:len(entries)] == entries
                        delta = fresh[len(entries):] if is_append else fresh
                        sequence += 1
                        await websocket.send_json({
                            "type": "transcript_append" if is_append else "transcript_reset",
                            "sequence": sequence, "meeting_id": run.meeting_id, "entries": delta,
                        })
                        entries = fresh
                    if service.status().get("state") in {"stopped", "error"} and summary_sent_for != run.meeting_id:
                        sequence += 1
                        await websocket.send_json({
                            "type": "summary_ready", "sequence": sequence,
                            "meeting_id": run.meeting_id,
                            "summary": str(run.transcript_path.with_suffix(".md")),
                        })
                        summary_sent_for = run.meeting_id
                await asyncio.sleep(0.4)
        except WebSocketDisconnect:
            return

    return app
