# Phase 0 Spike Report

Date: 2026-09-29  
Machine: Apple M5, Metal 4, macOS 27.0, arm64  
Status: Partial. Local ASR and model/document review are complete. Text-to-audio Gemini Live connectivity has now been verified from the user's terminal. Live microphone, interruption, long-session recovery, NON_BLOCKING tools, diarization, and Linux backend measurements remain pending.

## Scope and source review

Reviewed the [official Gemini Live API skill](https://github.com/google-gemini/gemini-skills/blob/main/skills/gemini-live-api-dev/SKILL.md), [Live API best practices](https://ai.google.dev/gemini-api/docs/live-api/best-practices), [session management](https://ai.google.dev/gemini-api/docs/live-api/session-management), [tool use](https://ai.google.dev/gemini-api/docs/live-api/tools), [thinking](https://ai.google.dev/gemini-api/docs/live-api/thinking), and [`cuppibla/live-dj`](https://github.com/cuppibla/live-dj). Its browser client sends 16 kHz PCM, plays 24 kHz PCM, stops playback locally using an RMS threshold, and also handles the server interruption event. Its raw SDK server runs microphone upload and Live receive concurrently and wraps `session.receive()` in an outer loop because it ends after each turn.

For Gemini Live, use `send_realtime_input` with 20–40 ms chunks; discard queued output when `server_content.interrupted` is true. The session config supports context compression and session resumption. Current Gemini 3.8 Live pricing is $0.005 per minute of audio input and $0.018 per minute of generated audio output. See the [official pricing table](https://ai.google.dev/gemini-api/docs/pricing#gemini-3.8-live-gemini-3.8-live-extended-thinking-and-gemini-3.1-flash-live-preview).

## M5 ASR benchmark

The audio is synthetic speech produced by macOS `say`, with one short sentence each in Chinese, English, and German. It is useful for a repeatable runtime smoke benchmark, not as real meeting audio or an acceptance sample. Each backend ran three times per single-language WAV; reported steady-state time is the median. The whisper.cpp server used the `ggml-large-v3-turbo-q5_0` quantized model (547 MB) and Metal. MLX used `mlx-community/whisper-large-v3-turbo`.

| Language | Audio | MLX median / RTF | whisper.cpp server median / RTF | Exact sentence recognized |
|---|---:|---:|---:|---|
| Chinese | 5.88 s | 0.430 s / 0.073 | 0.540 s / 0.092 | Yes, both |
| English | 6.39 s | 0.364 s / 0.057 | 0.502 s / 0.079 | Yes, both |
| German | 4.84 s | 0.392 s / 0.081 | 0.519 s / 0.107 | Yes, both |

Cold first-call latency in a fresh MLX process was 1.17–1.23 s for these short samples; subsequent calls in the same process were about 0.36–0.43 s. These timings include client/server overhead and model differences, so they are not a controlled accuracy comparison. Both backends were faster than audio duration on this M5.

Automatic language detection did not preserve all languages in the synthetic mixed clip: the direct full-clip run mostly returned German, and the Demo's 8-second windows recognized Chinese and German but omitted the English sentence. Fixed `--language zh|en|de` correctly transcribed each corresponding single-language sample. A self-recorded mixed-language sample is still required to decide how the application should handle code-switching.

Reproduce the measured backend calls with:

```bash
uv run python spikes/asr_bench.py /path/to/language.wav --language zh --backend mlx --repeat 3
whisper-server -m /path/to/ggml-large-v3-turbo-q5_0.bin --host 127.0.0.1 --port 8080
uv run python spikes/asr_bench.py /path/to/language.wav --language zh --backend whisper-server --model /path/to/ggml-large-v3-turbo-q5_0.bin --repeat 3
```

The HTTP ASR measurement used a local loopback server. Network transfer time is not included.

## Cost estimate

For one hour of continuous Gemini Live microphone input, the current published audio-input baseline is about **$0.30**. Add about **$0.018 per minute of generated agent speech**. For a 10-minute meeting with one minute of agent speech, the audio-only baseline is approximately **$0.068**. These are price-table calculations, not a measured invoice; Analyst text tokens, Google Search calls, and any other tool usage are additional.

Text-trigger mode avoids continuous Live audio input and its per-minute input cost, but has not been benchmarked for latency in this run.

## Gemini Live text-to-audio smoke run

The user ran `spikes/live_text_in_audio_out.py` from their configured terminal on 2026-09-29:

| Metric | Result |
|---|---:|
| Connection setup | 0.444 s |
| Request to first audio chunk | 0.837 s |
| Returned audio duration | 10.220 s |
| Output | `/tmp/live-text-response.wav` |

This verifies that the configured key can connect and return generated audio. The spike saves a WAV but does not play it; this is not an end-to-end audible playback or barge-in measurement.

## Gemini Live barge-in runs

The initial headphone run of `spikes/live_barge_in.py --minutes 2 --consent-to-stream` on 2026-09-29 lasted about 24.6 seconds:

| Metric | Result |
|---|---:|
| Local RMS stop callbacks | 10 |
| Server `interrupted` events | 2 |
| Dropped microphone chunks | 0 |
| Elapsed | 24.592 s |

The log confirmed generated audio playback, user speech reaching Gemini, local queue clearing, and server interruption handling. It also showed duplicate local triggers and unexpected Spanish/Portuguese recognition, so this first run did not establish reliable barge-in. The spike was then updated to count one local stop per continuous speech burst (re-armed after 350 ms below threshold) and to constrain response language to Chinese/English/German.

The adjusted headphone run lasted 16.889 seconds and recorded:

| Metric | Result |
|---|---:|
| Local RMS stop callbacks | 2 |
| Server `interrupted` events | 2 |
| Dropped microphone chunks | 0 |
| Elapsed | 16.889 s |

Both attempts to speak to the agent stopped local playback and were followed by a server interruption event. The first English prompt was transcribed as “Hey agent. Explain the trade-off.”; the follow-up was transcribed as “Aten”, so multilingual transcription is not established. This is an initial functional barge-in pass on headphones; speaker echo behavior and quantitative stop latency remain unmeasured. The monotonic timestamp is not a latency measurement.

## Remaining Phase 0 measurements

- Adjusted headphone barge-in run completed and is recorded above; basic local stop plus server `interrupted` was observed twice. Speaker echo behavior and stop latency remain unmeasured. `spikes/live_audio_passthrough.py` and `spikes/nonblocking_tool.py` have not yet been run. Long-session resumption remains unverified.
- The two microphone scripts require an explicit `--consent-to-stream` flag before they open the microphone.
- Keep the Gemini key local; do not paste it into chat. Run microphone tests only after participants agree. Start the barge-in trial with headphones, then separately repeat on speakers to measure acoustic echo false stops.
- No self-recorded 3–4 speaker sample or pyannote token was available. `spikes/diarization_bench.py` is still to be implemented; diarization accuracy and speed remain unmeasured.
- This is macOS hardware. whisper.cpp Vulkan and OpenVINO/NPU measurements require the target Omarchy machine and remain unmeasured.

## Recommendation

Use Gemini Live as a continuous listener that stays silent until the user directly addresses it with a question (for example, “Agent, what do you think?”); a separate wake-word-only utterance is not required. Keep local Whisper as the canonical transcript and `on_request` as the speech policy. Preserve text-trigger mode as the explicit privacy/cost alternative. Headphone barge-in now has an initial functional pass; long-session reliability and speaker echo behavior remain unverified.
