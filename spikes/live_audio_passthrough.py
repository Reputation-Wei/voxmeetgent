"""Stream the microphone to Gemini Live; measure latency, reconnects, and cost."""

import argparse
import asyncio
import os
import signal
import time
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
from google import genai
from google.genai import types


@dataclass
class Stats:
    started: float
    first_output_audio: float | None = None
    output_audio_bytes: int = 0
    reconnects: int = 0
    interruptions: int = 0


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=20)
    parser.add_argument("--consent-to-stream", action="store_true",
                        help="确认与会者同意将实时麦克风音频发送到 Gemini")
    parser.add_argument("--rms-threshold", type=float, default=0.02,
                        help="Local barge-in threshold (used in companion spike).")
    args = parser.parse_args()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("请先设置 GEMINI_API_KEY（或 GOOGLE_API_KEY）。")
    if args.minutes <= 0:
        parser.error("--minutes must be positive")
    if not args.consent_to_stream:
        parser.error("此测试会把麦克风实时发送到 Gemini；获得与会者同意后添加 --consent-to-stream")

    client = genai.Client(api_key=api_key)
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    reconnect = asyncio.Event()
    audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)
    stats = Stats(started=time.perf_counter())
    handle: str | None = None
    instruction = (
        "你在一场中英德混合的技术会议中。整场持续倾听并保持安静；只有有人明确点名 Agent 并提出问题时才简短发言。"
        "例如‘Agent，你怎么看？’或‘Hey Agent, what do you think?’无需先单独唤醒即可回答。"
        "不要长篇总结；用户开始说话时立刻停止并听用户说完。"
    )

    def mic_callback(indata: np.ndarray, frames: int, time_info: object, status: object) -> None:
        pcm = (np.clip(indata[:, 0], -1, 1) * 32767).astype("<i2").tobytes()

        def enqueue() -> None:
            if not audio_queue.full():
                audio_queue.put_nowait(pcm)

        loop.call_soon_threadsafe(enqueue)

    async def stream_for_one_connection() -> None:
        nonlocal handle
        config = types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            system_instruction=types.Content(parts=[types.Part(text=instruction)]),
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            context_window_compression=types.ContextWindowCompressionConfig(
                sliding_window=types.SlidingWindow(),
            ),
            session_resumption=types.SessionResumptionConfig(handle=handle),
        )
        async with client.aio.live.connect(model="gemini-3.8-live", config=config) as session:
            reconnect.clear()

            async def send_microphone() -> None:
                while not stop.is_set() and not reconnect.is_set():
                    chunk = await audio_queue.get()
                    await session.send_realtime_input(
                        audio=types.Blob(data=chunk, mime_type="audio/pcm;rate=16000"),
                    )

            async def receive_responses() -> None:
                empty_receives = 0
                while not stop.is_set() and not reconnect.is_set():
                    received = 0
                    async for response in session.receive():
                        received += 1
                        update = response.session_resumption_update
                        if update and update.resumable and update.new_handle:
                            handle = update.new_handle
                        if response.go_away:
                            print(f"goAway received; time_left={response.go_away.time_left}")
                            reconnect.set()
                            return
                        content = response.server_content
                        if content:
                            if content.interrupted:
                                stats.interruptions += 1
                                print("server interruption received")
                            if content.input_transcription and content.input_transcription.text:
                                print(f"heard: {content.input_transcription.text}")
                            if content.output_transcription and content.output_transcription.text:
                                print(f"agent: {content.output_transcription.text}")
                            if content.model_turn:
                                for part in content.model_turn.parts or []:
                                    audio = getattr(getattr(part, "inline_data", None), "data", None)
                                    if audio:
                                        stats.output_audio_bytes += len(audio)
                                        if stats.first_output_audio is None:
                                            stats.first_output_audio = time.perf_counter() - stats.started
                    if received == 0:
                        empty_receives += 1
                        if empty_receives >= 2:
                            reconnect.set()
                            return
                    else:
                        empty_receives = 0

            sender = asyncio.create_task(send_microphone(), name="send-microphone")
            receiver = asyncio.create_task(receive_responses(), name="receive-live")
            timer = asyncio.create_task(asyncio.sleep(args.minutes * 60), name="duration-limit")
            stop_waiter = asyncio.create_task(stop.wait(), name="stop-signal")
            await asyncio.wait({sender, receiver, timer, stop_waiter}, return_when=asyncio.FIRST_COMPLETED)
            for task in (sender, receiver, timer, stop_waiter):
                if not task.done():
                    task.cancel()
            await asyncio.gather(sender, receiver, timer, stop_waiter, return_exceptions=True)

    deadline = time.perf_counter() + args.minutes * 60
    previous_sigint = signal.getsignal(signal.SIGINT)
    loop.add_signal_handler(signal.SIGINT, stop.set)
    print(f"实时麦克风音频将发送到 Gemini，最长 {args.minutes:g} 分钟。直接问 'Agent, what do you think?' 请求回答；Ctrl+C 停止。")
    try:
        with sd.InputStream(samplerate=16_000, channels=1, dtype="float32", blocksize=320, callback=mic_callback):
            while not stop.is_set() and time.perf_counter() < deadline:
                await stream_for_one_connection()
                if stop.is_set() or time.perf_counter() >= deadline:
                    break
                stats.reconnects += 1
                print(f"resuming Live session #{stats.reconnects}")
    finally:
        stop.set()
        loop.remove_signal_handler(signal.SIGINT)
        signal.signal(signal.SIGINT, previous_sigint)
    elapsed_minutes = (time.perf_counter() - stats.started) / 60
    output_minutes = stats.output_audio_bytes / 48_000 / 60
    print(f"elapsed_minutes={elapsed_minutes:.3f}")
    print(f"first_output_audio_seconds={stats.first_output_audio}")
    print(f"output_audio_minutes={output_minutes:.4f}")
    print(f"reconnects={stats.reconnects}")
    print(f"server_interruptions={stats.interruptions}")
    print(f"audio_cost_baseline_usd={(elapsed_minutes * 0.005 + output_minutes * 0.018):.5f} (excludes text/tool costs)")


if __name__ == "__main__":
    asyncio.run(main())
