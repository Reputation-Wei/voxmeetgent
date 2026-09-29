"""Measure local RMS barge-in plus Gemini's server interrupted signal."""

import argparse
import asyncio
import os
import signal
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd
from google import genai
from google.genai import types


class PlaybackBuffer:
    """Small thread-safe PCM queue consumed by sounddevice's output callback."""

    def __init__(self) -> None:
        self._chunks: deque[np.ndarray] = deque()
        self._current: np.ndarray | None = None
        self._offset = 0
        self._lock = threading.Lock()
        self._speaking = False
        self.local_stops = 0
        self.server_stops = 0
        self.first_playback_at: float | None = None
        self.local_stop_pending = False

    def add(self, raw_pcm: bytes) -> None:
        pcm = np.frombuffer(raw_pcm, dtype="<i2").astype(np.float32) / 32768
        if pcm.size:
            with self._lock:
                self._chunks.append(pcm)

    def is_speaking(self) -> bool:
        with self._lock:
            return self._speaking or self._current is not None or bool(self._chunks)

    def clear(self, source: str) -> None:
        with self._lock:
            self._chunks.clear()
            self._current = None
            self._offset = 0
            self._speaking = False
            if source == "local":
                self.local_stops += 1
                self.local_stop_pending = True
            else:
                self.server_stops += 1

    def finish_local_speech(self) -> None:
        """Allow a later utterance to interrupt after the current one ends."""
        with self._lock:
            self.local_stop_pending = False

    def has_pending_local_stop(self) -> bool:
        with self._lock:
            return self.local_stop_pending

    def callback(self, outdata: np.ndarray, frames: int, time_info: object, status: object) -> None:
        outdata.fill(0)
        written = 0
        with self._lock:
            while written < frames:
                if self._current is None or self._offset >= len(self._current):
                    if not self._chunks:
                        break
                    self._current = self._chunks.popleft()
                    self._offset = 0
                count = min(frames - written, len(self._current) - self._offset)
                outdata[written : written + count, 0] = self._current[self._offset : self._offset + count]
                written += count
                self._offset += count
            if written:
                self._speaking = True
                if self.first_playback_at is None:
                    self.first_playback_at = time.perf_counter()
            elif not self._chunks:
                self._speaking = False


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=2)
    parser.add_argument("--rms-threshold", type=float, default=0.02)
    parser.add_argument("--consent-to-stream", action="store_true",
                        help="确认与会者同意将实时麦克风音频发送到 Gemini")
    args = parser.parse_args()
    if args.minutes <= 0:
        parser.error("--minutes must be positive")
    if not args.consent_to_stream:
        parser.error("此测试会把麦克风实时发送到 Gemini；获得与会者同意后添加 --consent-to-stream")
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("请先设置 GEMINI_API_KEY（或 GOOGLE_API_KEY）。")

    client = genai.Client(api_key=api_key)
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    microphone: asyncio.Queue[bytes] = asyncio.Queue(maxsize=100)
    playback = PlaybackBuffer()
    local_stop_times: list[float] = []
    dropped_chunks = 0
    below_threshold_since: float | None = None
    silence_reset_seconds = 0.35

    def mic_callback(indata: np.ndarray, frames: int, time_info: object, status: object) -> None:
        nonlocal dropped_chunks, below_threshold_since
        data = indata[:, 0].copy()
        rms = float(np.sqrt(np.mean(data * data)))
        now = time.perf_counter()
        if rms >= args.rms_threshold:
            below_threshold_since = None
        elif below_threshold_since is None:
            below_threshold_since = now
        elif now - below_threshold_since >= silence_reset_seconds:
            playback.finish_local_speech()

        if (rms >= args.rms_threshold and playback.is_speaking()
                and not playback.has_pending_local_stop()):
            playback.clear("local")
            local_stop_times.append(now)
            loop.call_soon_threadsafe(print, f"local barge-in RMS={rms:.4f}; playback stopped immediately")
        pcm = (np.clip(data, -1, 1) * 32767).astype("<i2").tobytes()

        def enqueue() -> None:
            nonlocal dropped_chunks
            if microphone.full():
                dropped_chunks += 1
            else:
                microphone.put_nowait(pcm)

        loop.call_soon_threadsafe(enqueue)

    config = types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        system_instruction=types.Content(parts=[types.Part(text=(
            "你是多语会议助手。会议中持续倾听，但必须保持安静；只有有人明确点名你并提出问题时才简短回答。"
            "例如‘Agent，你怎么看？’或‘Hey Agent, what do you think?’都直接回答，不要求先单独说唤醒词。"
            "仅用中文、英语或德语交流；默认跟随说话者的语言，不要自行切换到其他语言。"
            "说话者插话时立即停下并听他们说完。回答保持简短，便于被自然打断。"
        ))]),
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
    )
    previous_sigint = signal.getsignal(signal.SIGINT)
    loop.add_signal_handler(signal.SIGINT, stop.set)
    started = time.perf_counter()
    print("实时麦克风音频将发送到 Gemini。戴上耳机：直接问 'Agent, what do you think about this trade-off?'，再在它回答时插话。Ctrl+C 结束。")
    try:
        with (
            sd.InputStream(samplerate=16_000, channels=1, dtype="float32", blocksize=320, callback=mic_callback),
            sd.OutputStream(samplerate=24_000, channels=1, dtype="float32", blocksize=480, callback=playback.callback),
        ):
            async with client.aio.live.connect(model="gemini-3.8-live", config=config) as session:
                async def send_microphone() -> None:
                    while not stop.is_set():
                        chunk = await microphone.get()
                        await session.send_realtime_input(
                            audio=types.Blob(data=chunk, mime_type="audio/pcm;rate=16000"),
                        )

                async def receive_live() -> None:
                    while not stop.is_set():
                        got = 0
                        async for response in session.receive():
                            got += 1
                            content = response.server_content
                            if content:
                                if content.interrupted:
                                    playback.clear("server")
                                    print("server interrupted=true; playback queue cleared")
                                if content.input_transcription and content.input_transcription.text:
                                    print(f"you: {content.input_transcription.text}")
                                if content.output_transcription and content.output_transcription.text:
                                    print(f"agent: {content.output_transcription.text}")
                                if content.model_turn:
                                    for part in content.model_turn.parts or []:
                                        audio = getattr(getattr(part, "inline_data", None), "data", None)
                                        if audio:
                                            playback.add(audio)
                        if got == 0:
                            return

                sender = asyncio.create_task(send_microphone())
                receiver = asyncio.create_task(receive_live())
                timer = asyncio.create_task(asyncio.sleep(args.minutes * 60))
                stop_waiter = asyncio.create_task(stop.wait())
                await asyncio.wait({sender, receiver, timer, stop_waiter}, return_when=asyncio.FIRST_COMPLETED)
                for task in (sender, receiver, timer, stop_waiter):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(sender, receiver, timer, stop_waiter, return_exceptions=True)
    finally:
        stop.set()
        loop.remove_signal_handler(signal.SIGINT)
        signal.signal(signal.SIGINT, previous_sigint)
    print(f"elapsed_seconds={time.perf_counter() - started:.3f}")
    print(f"local_barge_ins={playback.local_stops}")
    print(f"server_interruptions={playback.server_stops}")
    print(f"dropped_microphone_chunks={dropped_chunks}")
    if local_stop_times:
        print(f"last_local_stop_monotonic={local_stop_times[-1]:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
