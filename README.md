# MeetAgent

[English](README.en.md) | 中文

MeetAgent 是一款本地优先的多语言会议助手：在 Apple Silicon Mac 上使用本机 Whisper 转写会议，持续保存原始音频和逐字稿；可选连接 Gemini Live，让 Agent 持续倾听：被点名提问时回答，说出研究、事实核查、逻辑检查或总结关键词时自动搜索并播报结论。项目也提供本机 Web GUI，以及 Omarchy / Quickshell 状态栏插件。

> 开会前请告知与会者正在录音和转写，并取得必要同意。只有在另外明确授权后，MeetAgent 才会把整场音频发送到 Gemini Live；会后把逐字稿发送给 Gemini 生成纪要也是单独选择，默认关闭。

## 功能

- 本地保存会议 WAV、可读 TXT、结构化 JSONL 和 Markdown 纪要。
- Apple Silicon macOS 默认使用本机 MLX Whisper `large-v3`；模型首次使用时自动下载。
- 支持自动识别或指定中文、英语、德语。
- 声纹辅助标记发言人，并支持会后修正说话人。
- Gemini Live 可持续倾听；Agent 默认安静，被点名或听到关键词（“有没有相关研究”“这是真的吗”“总结一下”等，中英德）时回答，回答可被插话打断。
- 本机 Web GUI 可开始/结束会议、查看转写、修正说话人和访问会议文件。
- Omarchy Quickshell 插件提供状态栏状态、计时、快速录音控制和 GUI 入口。

识别结果和说话人标签可能出错。重叠讲话、远场收音和噪声无法保证逐字准确；原始 WAV 是复核依据。功能状态和已知限制见 [PLAN.md](PLAN.md) 与 [README-demo.md](README-demo.md)。

## 环境要求

- macOS Apple Silicon、Python 3.12、[`uv`](https://docs.astral.sh/uv/) 和 `ffmpeg`。
- `GEMINI_API_KEY` 可选：本地录音/转写不需要它；Gemini Live 和 Gemini 纪要需要它。
- Linux 支持状态及 Omarchy 插件安装说明见 [`omarchy-plugin/README.md`](omarchy-plugin/README.md)。Linux Whisper 目前需要用户提供 whisper.cpp GGML 模型，通过 `MEETAGENT_WHISPER_MODEL` 或 `~/.config/meetagent/config.toml` 的 `[audio] model_path` 指定。

### Linux GPU 转写（Intel Arc / Vulkan）

PyPI 的 `pywhispercpp` 轮子只含 CPU 后端。在有 Vulkan 驱动的 Linux（如 Omarchy + Intel Arc B390）上，用脚本在项目 `.venv` 内重新编译带 Vulkan 的版本：

```bash
./scripts/build-whisper-vulkan.sh
uv run meetagent doctor   # 应显示 “whisper.cpp GPU: Vulkan 后端已编译”
```

需要 `vulkan-intel`、`shaderc`、`cmake`、`gcc`；系统未装 `vulkan-headers`/`spirv-headers` 时，脚本会把头文件下载到 `~/.cache/meetagent/vulkan-sdk`，无需 root。B390 上 60 秒中文录音的转写从 CPU 的 19.3 秒降到 3.8 秒（large-v3-turbo），完整 large-v3 为 3.3 秒。`uv sync --reinstall` 或重建 `.venv` 后需重新运行脚本。

macOS 安装依赖并同步项目：

```bash
brew install uv ffmpeg
git clone https://github.com/Reputation-Wei/voxmeetgent.git
cd voxmeetgent
uv sync
```

MLX Whisper 模型与说话人 embedding 模型会在首次使用时下载并缓存；无需手动预装 Whisper 权重。

## 启动图形界面

```bash
uv run meetagent gui
```

服务只监听本机 `127.0.0.1:8765` 并打开浏览器。关闭运行服务的终端会停止 GUI。界面默认只启动本地录音/转写；开启 Gemini Live 前必须确认与会者同意将整场音频发送到 Gemini。会后纪要上传有独立同意选项。

若只需启动本机服务而不自动打开浏览器：

```bash
uv run meetagent gui --no-open
```

## 命令行录音

设置会议输出目录和识别语言：

```bash
uv run meetagent demo --out ~/Meetings --language zh
```

`--language` 可用 `auto`、`zh`、`en`、`de`。没有 Gemini key 时，仍会本地录音和转写。停止会议按 `Ctrl+C`；GUI 则使用停止按钮。

### Gemini Live 与纪要

需要 Gemini 功能时，在当前终端设置 API key：

```bash
export GEMINI_API_KEY="你的 API key"
```

Live 模式会上传整场麦克风音频，需要明确添加同意选项：

```bash
uv run meetagent demo --out ~/Meetings --language zh --consent-to-stream
```

未提供 `--consent-to-stream` 时，CLI 只运行本地录音和转写。CLI 下 Gemini 会后纪要默认可用（前提是提供 key）；若要确保纪要不上传，请不设置 Gemini key，或改用 GUI 并保持“允许将逐字稿发送给 Gemini 生成纪要”关闭。不要将 API key 写入源码、提交到 Git 或分享给他人。

## 输出文件

默认写到 `~/Meetings`，也可用 `--out` 指定目录。每场会议会保存：

- `.wav`：本地原始录音。
- `.txt`：可读逐字稿，识别内容逐段写入。
- `.jsonl`：包含时间、说话人及识别信息的逐条记录。
- `.md`：会后纪要；没有 Gemini 生成许可或请求不可用时，会保存本地逐字稿型 Markdown。

可选把 Markdown 纪要额外复制到 Obsidian vault；会议音频和原始转写仍保存在输出目录：

```bash
uv run meetagent demo --out ~/Meetings \
  --obsidian-vault ~/Documents/Obsidian \
  --obsidian-folder Meetings
```

## 检查本机依赖

```bash
uv run meetagent doctor
```

查看所有命令：

```bash
uv run meetagent --help
```

## Omarchy / Quickshell

插件源文件和完整安装说明位于 [`omarchy-plugin/`](omarchy-plugin/README.md)。插件连接本机 GUI 服务，因此先在 Omarchy 运行：

```bash
meetagent gui --no-open
```

之后按插件 README 安装并启用 bar-widget。插件的 Omarchy 真机加载和录音链路仍需在 Omarchy 主机验收。

## 开发

```bash
uv sync --group dev
uv run meetagent --help
```

代码在 `src/meetagent/`，设计记录和阶段进度在 `docs/` 与 [PLAN.md](PLAN.md)。在提交会议数据前请确认 `.gitignore` 仍排除了密钥、音频、逐字稿和本地运行缓存。
