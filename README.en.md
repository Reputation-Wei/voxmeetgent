# MeetAgent

[English](README.en.md) | [中文](README.md)

MeetAgent is a local-first multilingual meeting assistant. It transcribes meetings on Apple Silicon Macs with local Whisper while saving the original audio and transcript. You can optionally connect Gemini Live so the agent can listen throughout a meeting and speak only when explicitly addressed. The project also includes a local web GUI and an Omarchy / Quickshell status-bar plugin.

> Before a meeting, tell participants that it will be recorded and transcribed, and obtain the required consent. MeetAgent sends the full meeting audio to Gemini Live only after you explicitly enable that option. Sending the transcript to Gemini for a summary is a separate choice.

## Features

- Saves meeting audio as WAV, readable transcripts as TXT, structured entries as JSONL, and meeting notes as Markdown.
- Uses local MLX Whisper `large-v3` by default on Apple Silicon macOS. Model weights are downloaded on first use.
- Supports automatic language detection or Chinese, English, and German.
- Uses voice embeddings to help label speakers; speaker names can be corrected after a meeting.
- Gemini Live can listen continuously. The agent stays quiet until explicitly addressed, and you can interrupt spoken replies.
- The local web GUI can start and stop meetings, show transcripts, correct speaker labels, and access meeting files.
- The Omarchy Quickshell plugin shows meeting status and duration, provides recording controls, and links to the GUI.

Transcription and speaker labels can be wrong. Overlapping speech, distant microphones, and noise prevent guaranteed word-for-word accuracy. Keep the original WAV as the source for review. See [PLAN.md](PLAN.md) and [README-demo.md](README-demo.md) for project status and limitations.

## Requirements

- macOS on Apple Silicon, Python 3.12, [`uv`](https://docs.astral.sh/uv/), and `ffmpeg`.
- `GEMINI_API_KEY` is optional. Local recording and transcription do not need it; Gemini Live and Gemini-generated summaries do.
- See [`omarchy-plugin/README.md`](omarchy-plugin/README.md) for Linux and Omarchy plugin details. Linux transcription currently requires a user-provided whisper.cpp GGML model.

Install the dependencies and sync the project on macOS:

```bash
brew install uv ffmpeg
git clone https://github.com/Reputation-Wei/voxmeetgent.git
cd voxmeetgent
uv sync
```

MLX Whisper and speaker embedding models are downloaded and cached on first use. You do not need to install Whisper weights manually.

## Launch the GUI

```bash
uv run meetagent gui
```

The service listens only on `127.0.0.1:8765` and opens a browser. Closing the terminal running the service stops the GUI. The interface defaults to local recording and transcription. Before enabling Gemini Live, confirm that participants agree to send the full meeting audio to Gemini. Transcript upload for meeting summaries has its own consent option.

To start the local service without opening a browser automatically:

```bash
uv run meetagent gui --no-open
```

## Record from the command line

Choose an output directory and transcription language:

```bash
uv run meetagent demo --out ~/Meetings --language zh
```

`--language` accepts `auto`, `zh`, `en`, or `de`. Local recording and transcription still work without a Gemini API key. Press `Ctrl+C` to stop a CLI meeting; in the GUI, use the Stop button.

### Gemini Live and summaries

Set your API key in the current terminal when you want to use Gemini:

```bash
export GEMINI_API_KEY="your API key"
```

Live mode sends the full microphone audio stream to Gemini and requires the explicit consent flag:

```bash
uv run meetagent demo --out ~/Meetings --language zh --consent-to-stream
```

Without `--consent-to-stream`, the CLI runs local recording and transcription only. In the current CLI implementation, Gemini summary generation is enabled when an API key is available. To ensure transcript text is not uploaded for summaries, leave the key unset or use the GUI and keep “Allow transcript upload to Gemini for a summary” turned off. Never put your API key in source code or commit it to Git.

## Output files

Meetings are saved to `~/Meetings` by default. Use `--out` to choose another directory. Each meeting produces:

- `.wav`: original local audio recording.
- `.txt`: readable transcript, written incrementally as speech is recognized.
- `.jsonl`: structured entries with timestamps, speaker labels, and recognition details.
- `.md`: meeting notes. If Gemini summary generation is unavailable or not enabled, a local Markdown transcript-style note is saved instead.

You can also copy the Markdown summary into an Obsidian vault. Audio and original transcripts remain in the output directory:

```bash
uv run meetagent demo --out ~/Meetings \
  --obsidian-vault ~/Documents/Obsidian \
  --obsidian-folder Meetings
```

## Check local dependencies

```bash
uv run meetagent doctor
```

List all commands:

```bash
uv run meetagent --help
```

## Omarchy / Quickshell

Plugin source and installation instructions are in [`omarchy-plugin/`](omarchy-plugin/README.md). The plugin connects to the local GUI service, so start it on Omarchy first:

```bash
meetagent gui --no-open
```

Then follow the plugin README to install and enable the bar widget. Loading the plugin and recording from an Omarchy host still need target-machine validation.

## Development

```bash
uv sync --group dev
uv run meetagent --help
```

Source code is in `src/meetagent/`; design decisions and project progress are documented in `docs/` and [PLAN.md](PLAN.md). Before committing, confirm `.gitignore` excludes API keys, meeting audio, transcripts, and local runtime caches.
