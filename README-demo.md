# MeetAgent 最小 Demo（Phase D）

这个 Demo 在 Apple Silicon Mac 上用本地 Whisper 转写会议音频，同时让 Gemini Live 持续倾听；Live 默认静默，用户点名提问时语音回答。研究、事实核查、逻辑检查和总结关键词无需点名，由 Gemini 搜索分析后播报，结束时生成 Markdown 纪要和 JSONL 转写。VAD 切出的每段语音会用本地 sherpa-onnx 声纹模型匹配说话人；未注册者会在本场会议内临时分配编号。

实时 Live 收发与可打断播放已接入 Demo；仍需在用户的 M5 和实际会议环境验证端到端语音行为。持续倾听模式会把会议麦克风音频持续发送到 Gemini，启动前需明确提示并取得与会者同意；本地 Whisper 继续负责正式逐字稿。

## 安装

需要 macOS Apple Silicon、Python 3.12、`uv` 和 `ffmpeg`。实时麦克风模式当前只支持 Apple Silicon Mac。

```bash
brew install uv ffmpeg
uv sync
```

启动本机 GUI：

```bash
uv run meetagent gui
```

命令会启动仅监听 `127.0.0.1` 的本地服务并打开浏览器；退出运行该命令的终端会关闭 GUI 服务。默认开始本地录音，不上传音频。勾选 Gemini Live 后，必须再勾选与会者已同意整场音频上传，开始按钮才会向后端提交 Live 模式。会后逐字稿上传是单独选项，默认关闭；未授权时会生成本地模板纪要。会议文件仍保存在本机指定目录。

macOS M5 默认使用精度优先的完整 Whisper `large-v3`（MLX 转换版 `mlx-community/whisper-large-v3-mlx`），首次运行自动下载约 3.1 GB 权重。OpenAI 将 `large-v3-turbo` 描述为更快、准确度有轻微下降的优化版；需要优先速度时可设置 `MEETAGENT_WHISPER_MODEL_ID=mlx-community/whisper-large-v3-turbo`。两种模型都在本机运行，无需把语音发给 Whisper 云服务。

第一次转写还会从 sherpa-onnx 下载说话人 embedding 模型到 `~/.cache/meetagent/models/`。可用 `MEETAGENT_SPEAKER_MODEL` 指定已有的 ONNX 模型路径。未注册声纹的短句或不够清晰的发言会保守标为“说话人未知”；同一个人注册的声音可用 `meetagent enroll` 增加姓名匹配机会，但必须是清晰的单人样本。

每位固定与会者可录制 5–10 秒、安静环境下的单人 16-bit PCM WAV 注册声纹；可用多段录音重复注册同一姓名，以覆盖自然音色变化。对自动标签有疑问时，先从对应 JSONL 找该句的 `timestamp` 秒数，再同时修正可读稿与结构化稿：

```bash
uv run meetagent speaker-correct --transcript ~/Meetings/meeting-时间戳.txt --time 123.45 --name "张三"
```

修正会在 JSONL 里记录原说话人和修正标记。

实时麦克风可选启用 WebRTC 声学回声消除：

```bash
uv sync --extra aec
```

默认会尝试启用 AEC；没安装 extra 时会提示并继续以未处理麦克风音频运行。可用 `--no-aec` 临时关闭，或在 `~/.config/meetagent/config.toml` 设置：

```toml
[audio]
aec_enabled = true
aec_delay_ms = 20
```

`aec_delay_ms` 是输入/输出设备和扬声器的参考延迟估计，默认 20 ms；本机 M5 的低音量参考信号对比中，20–35 ms 优于 50 ms。房间、设备和缓冲不同会影响效果，需要按实际设备调校。

## 设置 Gemini API key

在 [Google AI Studio](https://aistudio.google.com/apikey) 创建 Gemini Developer API key，然后在当前终端设置：

```bash
export GEMINI_API_KEY="你的 API key"
```

API key 只从环境变量读取。没有 key 时，麦克风会议仍可本地录音和 Whisper 转写，但不会连接 Live 或上传音频；设置 key 后，实时倾听还必须加 `--consent-to-stream`。Gemini 不可用时仍会按本地逐字稿生成 Markdown 纪要。

## 运行

实时麦克风：

```bash
uv run meetagent demo --out ~/Meetings --language zh
```

实时模式需要确认与会者同意整场音频发送到 Gemini Live，因此必须显式添加授权选项：

```bash
uv run meetagent demo --out ~/Meetings --language zh --consent-to-stream
```

可用 `--intervention` 选择策略：`on_request`（默认，仅在识别到“Agent，你怎么看？”或英文/德文对应呼叫后回答）、`silent`（完全不播报）或 `proactive`。除 `silent` 外，点名提问和关键词（相关研究、事实核查、逻辑检查、总结，中英德，见 `src/meetagent/keywords.yaml`）都会触发播报；Agent 不会在无人请求时主动发言（后续“举手”机制需人批准，见 PLAN.md Phase 8）。可在 `~/.config/meetagent/config.toml` 配置，命令行参数优先：

```toml
[live]
intervention = "on_request"
rms_threshold = 0.02
```

例如临时静默运行：

```bash
uv run meetagent demo --out ~/Meetings --consent-to-stream --intervention silent
```

可选地增加 `--analyze-claims`，每分钟把最近两分钟的转写文字额外发给 Gemini Flash，静默抽取可核查陈述并保存 Finding；此选项默认关闭，不会语音播报：

```bash
uv run meetagent demo --out ~/Meetings --language zh --consent-to-stream --analyze-claims
```

麦克风音频会送入本地 VAD/Whisper；只有设置 key 并确认同意后才另外发送到 Gemini Live。Live 生成回答的唤醒依据由本地 Whisper 的完整 `large-v3` 识别结果提供：只有识别到“Agent，你怎么看？”才会发送点名文本并开放当前回合播放，减少对 Gemini 自己识别唤醒词的依赖。Agent 播报时可直接插话。研究、事实核查、逻辑检查和总结关键词无需点名：分析在后台运行，不阻塞本地转写，完成后用一两句口语播报结论，完整结果和来源写入终端与逐字稿。

`--language` 支持 `zh`、`en`、`de` 和 `auto`（默认）。以一种语言为主的会议可指定主语言；中英德频繁混说时建议用 `auto`，固定语言可能降低其他语言片段的识别效果。当前对 Whisper 输出增加了短音频/近静音筛选、段级静音概率/平均置信度/重复度过滤，以减少幻听；这会带来少量漏掉轻声短词的风险，阈值可再根据你的真实录音调校。完整 `large-v3` 仍无法保证逐字零错误：多人同时讲话、远场收音和噪声都可能导致漏字或串字；当前 mono 单声道输入无法恢复重叠人声各自的完整内容。原始 WAV 是核对依据，TXT/JSONL 保存模型实际识别结果。

幻听抑制阈值可按录音情况用环境变量调整。默认偏严格；若真实语音被漏掉，可在启动会议前小步降低相应门限：`MEETAGENT_WHISPER_NO_SPEECH=0.65`、`MEETAGENT_WHISPER_LOGPROB=-1.2`、`MEETAGENT_WHISPER_COMPRESSION=2.4`。更高的静音门限和更低的 logprob 门限会保留更多片段，也可能让幻听回来。

Linux 需设置 `MEETAGENT_WHISPER_MODEL` 或 config.toml 的 `[audio] model_path` 指向本地 GGML 模型文件；Demo 当前不会自动下载 Linux 模型。Intel Arc 等 GPU 加速见 README 的 “Linux GPU 转写”。

离线 WAV：

```bash
uv run meetagent demo --file tests/fixtures/trilingual.wav --out ~/Meetings
```

离线模式也可以指定语言，例如 `--language de`。

仅用本地 VAD + Whisper 处理 WAV，不连接 Gemini，也不生成云端回答：

```bash
uv run meetagent transcribe --file meeting.wav --out ~/Meetings --language auto
```

该命令生成原始 WAV、可读 `.txt`、结构化 `.jsonl` 和本地逐字稿 `.md`。

## 会后纪要（Phase 5）

结束会议后默认用 `gemini-3.8-flash` 生成中文 Markdown 纪要；长会议逐字稿会按发言行切块提取后再合并。摘要包含摘要、议题、决议、待办（负责人/截止日期）、事实核查、研究方向、未解决问题，以及 Agent 回答和来源。没有明确说出的决定或任务必须列为“未记录”，不会补造。若没有 Gemini key 或请求失败，仍本地保存逐字稿型 Markdown，并明确标注未生成模型归纳。

指定纪要语言：

```bash
uv run meetagent demo --out ~/Meetings --summary-language en
uv run meetagent demo --out ~/Meetings --summary-language de
```

可选把 Markdown 纪要额外复制到 Obsidian vault（原 WAV、TXT、JSONL 仍保存在 `--out`）：

```bash
uv run meetagent demo --out ~/Meetings --summary-language zh \\
  --obsidian-vault ~/Documents/Obsidian --obsidian-folder Meetings
```

也可在 `~/.config/meetagent/config.toml` 设为默认：

```toml
[output]
directory = "~/Meetings"
obsidian_vault = "~/Documents/Obsidian"
obsidian_folder = "Meetings"
```

Obsidian 写入要求目录内存在 `.obsidian/`。纪要先原子写入 `--out` 本地目录，再复制进 vault；vault 写入失败时，本地 Markdown 仍保留并会显示错误。

会前注册与会者声纹。准备一个至少 2 秒的单人 16-bit PCM WAV，建议录制 5–10 秒清晰语音：

```bash
uv run meetagent enroll --name "张三" --file zhangsan.wav
```

注册数据保存在 `~/.local/share/meetagent/speakers.json`，仅存声纹向量，不保存此注册 WAV。未注册说话人只有在多段清晰语音相互确认后才会从 `Speaker 1` 开始编号；未确认时标记为“说话人未知”。

离线文件须为 16-bit PCM WAV；程序会转换采样率并混合成单声道。按 `Ctrl+C` 结束麦克风会议并生成纪要。离线模式会处理到文件末尾后生成输出。

启动时会先在输出目录创建 `meeting-时间戳.txt` 和对应的 `.jsonl`。麦克风 WAV 每个回调立即写入并更新 WAV 长度头，每秒 `fsync`；即使 ASR 忙，采集帧也会排队而不会因 4 秒队列上限被静默丢弃。每识别出一段文字就立即写入并 `fsync` TXT/JSONL；Live 转写只作兜底，稍后本地结果会替换匹配项。逐字稿本地 ASR 始终使用未经过 AEC 修改的原始麦克风帧；AEC 只作用于 Live 上传和插话检测。Agent 回复也写入同一时间线。原始 WAV 是需要复核时的依据，`.txt` 为可读逐字稿，`.jsonl` 保留语言、时间和来源。识别和说话人标注仍可能出错；重叠发言不能靠单路麦克风与 Whisper 保证逐字无误，重要会议应保留原始音频并人工复核。

## 已知限制

- macOS 使用 MLX Whisper；Linux 离线/麦克风转写使用 pywhispercpp，但需要用户提供 GGML 模型路径。Linux 音频设备与 Live 实时路径尚未在目标机验证。
- 说话人身份按 VAD 语音段的声纹相似度匹配；未注册者仅在单场会议内聚类。多人同时说话时不会把重叠语音拆开。注册声纹存储在本机 JSON 文件中。
- 本仓库的 `trilingual.wav` 是合成测试信号，便于验证音频读取流程；它不是语音，也不满足计划中“自录三语样例”的验收要求。请用真实中、英、德混说录音进一步试用。
- 关键词按转写片段匹配；超长发言可能要等到 VAD 检出停顿后才显示。此 Demo 还没有达到“5 秒内转写”的实机验收。
- Live 使用上下文压缩和 session resumption 处理服务端连接轮换；连接错误按 1、2、4、8、16、30 秒退避重试，长时间会议恢复仍待实机验证。
- 已接入可选 WebRTC 声学回声消除；AEC 使用从输出音频生成的参考信号，清理后的音频用于识别、Live 上传和插话门限，WAV 保留原始麦克风录音。延迟估计、扬声器环境效果和误触发率仍待 M5 实测；建议先用耳机试用。
- 实时模式持续将麦克风音频发送到 Gemini。按当前费率，音频输入约 $0.30/小时，生成语音另计；会前须告知所有与会者并取得必要同意。
- 搜索 grounding 的来源链接依赖 API 返回的 grounding metadata。
- API 花费、识别延迟和 10 分钟会议成本尚未在目标 M5 设备上实测；不能据此承诺计划中的延迟验收。
- 原始会议音频以 WAV 文件保存在输出目录；开始前请告知与会者并取得必要同意。

开始前请告知所有与会者会议正在转写，并取得必要同意。
