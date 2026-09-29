"""Measure Gemini Live connection and first-audio latency for text input."""

import argparse
import asyncio
import os
import time
import wave
from pathlib import Path

from google import genai
from google.genai import types


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="请用中文简短回答：实时会议助手应如何避免打断与会者？")
    parser.add_argument("--out", type=Path, default=Path("/tmp/live-text-response.wav"))
    args = parser.parse_args()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("请先设置 GEMINI_API_KEY（或 GOOGLE_API_KEY）。")

    client = genai.Client(api_key=api_key)
    connect_start = time.perf_counter()
    audio_chunks: list[bytes] = []
    first_audio_seconds: float | None = None
    async with client.aio.live.connect(
        model="gemini-3.8-live",
        config=types.LiveConnectConfig(response_modalities=[types.Modality.AUDIO]),
    ) as session:
        connected_at = time.perf_counter()
        await session.send_client_content(
            turns=types.Content(role="user", parts=[types.Part(text=args.prompt)]),
            turn_complete=True,
        )
        request_started = time.perf_counter()
        while True:
            got_messages = 0
            content = None
            async for response in session.receive():
                got_messages += 1
                content = response.server_content
                if response.data:
                    chunks = [response.data]
                else:
                    model_turn = getattr(content, "model_turn", None) if content else None
                    chunks = [
                        part.inline_data.data
                        for part in (getattr(model_turn, "parts", None) or [])
                        if getattr(part, "inline_data", None) is not None
                        and getattr(part.inline_data, "data", None)
                    ]
                if chunks:
                    if first_audio_seconds is None:
                        first_audio_seconds = time.perf_counter() - request_started
                    audio_chunks.extend(chunks)
                if content and content.turn_complete:
                    break
            if got_messages == 0 or (content and content.turn_complete):
                break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(args.out), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24_000)
        wav.writeframes(b"".join(audio_chunks))
    print(f"connect_seconds={connected_at - connect_start:.3f}")
    print(f"first_audio_seconds={first_audio_seconds}")
    print(f"output_audio_seconds={sum(map(len, audio_chunks)) / 48_000:.3f}")
    print(f"saved_audio={args.out}")


if __name__ == "__main__":
    asyncio.run(main())
