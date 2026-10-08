"""`meetagent follow` against a fake local service."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from rich.console import Console

from meetagent.ui.follow import run_follow

ENTRIES = [
    {"time": "00:02", "speaker": "张三", "text": "ImageNet 有一千万张图片"},
    {"time": "00:06", "speaker": "张三", "text": "Agent 你怎么看"},
    {"time": "00:15", "speaker": "Agent", "text": "## 回答\n完整数据集约 **1400 万**张。"},
]


def _server(script: list[tuple[str, int]]):
    """script: per status poll, (state, number of transcript entries visible)."""
    calls = {"status": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, payload):
            body = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/api/bootstrap":
                return self._send({"token": "t"})
            assert self.headers["X-MeetAgent-Token"] == "t"
            step = script[min(calls["status"] - 1, len(script) - 1)]
            if self.path == "/api/status":
                calls["status"] += 1
                state, _ = script[min(calls["status"] - 1, len(script) - 1)]
                return self._send({
                    "state": state, "meeting_id": "m1", "reply": "text", "output_dir": "/tmp/m",
                    "transcript": "/tmp/m/meeting-m1.txt", "summary_path": "/tmp/m/meeting-m1.md",
                })
            offset = int(self.path.split("offset=")[1].split("&")[0])
            visible = ENTRIES[: step[1]]
            return self._send({"entries": visible[offset:], "total": len(visible)})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_follow_prints_each_line_once_and_reports_the_end() -> None:
    server = _server([("recording", 1), ("recording", 2), ("recording", 3), ("stopped", 3), ("stopped", 3)])
    console = Console(record=True, width=100)
    try:
        run_follow(server.server_address[1], poll_seconds=0, console=console, max_polls=5)
    finally:
        server.shutdown()
    out = console.export_text()
    assert out.count("ImageNet 有一千万张图片") == 1
    assert "[00:06] 张三: Agent 你怎么看" in out
    assert "Agent · 00:15" in out and "1400 万" in out and "**" not in out  # Markdown rendered
    assert out.count("会议已结束") == 1
    assert "/tmp/m/meeting-m1.agent.md" in out and "文字回答" in out


def test_follow_waits_when_service_is_down() -> None:
    console = Console(record=True, width=100)
    run_follow(9, poll_seconds=0, console=console, max_polls=1)  # nothing listens on port 9
    assert "等待 MeetAgent 服务启动" in console.export_text()
