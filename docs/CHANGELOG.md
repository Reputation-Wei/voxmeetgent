# Changelog

## Phase 0 continuation — 2026-09-29

### 已完成

- 按 `cuppibla/live-dj` 与 Gemini Live 官方资料复核交互实现，更新 `PLAN.md`：增加本地 RMS/VAD 即时停播、服务端 `interrupted` 清队列、并发收发及持续重启 `session.receive()` 的要求。
- 在 Apple M5 上用 MLX Whisper 与 whisper.cpp Metal server 对中文、英语、德语合成单语短句各测 3 次；固定语言结果都逐字正确，常驻中位延迟约 0.36–0.54 秒。详见 `docs/spike-report.md`。
- 准备文本触发 Live、实时音频、barge-in 与 NON_BLOCKING 工具 spike；Python 编译及 CLI 参数帮助检查通过。

### 尚待实测

- 初次执行时 shell 未配置 Gemini API key；之后用户在本机终端运行了文本到语音和耳机打断 spike，结果详见 `docs/spike-report.md`。长会话恢复和 NON_BLOCKING 工具仍待测。
- 缺少自录混说及多人带标注音频，真实会议 ASR 与说话人分离仍未验收；diarization benchmark 脚本待实现。
- Linux Vulkan/OpenVINO 需要在目标 Linux 设备测量。

## Phase 1 start — 2026-09-29

### 已完成

- 新增 `docs/ARCHITECTURE.md` 与 ADR-001～009，明确组件边界、事件流、会话状态、实时语音打断和同意要求；未经实测的选择标为暂定。
- 新增 typed 配置和事件模型、带有界订阅队列的 asyncio EventBus，以及音频、ASR、说话人分离、存储、Agent、触发器与 UI 接口骨架。
- 新增 `meetagent doctor`；检查输出目录写权限、输入设备列表、Metal 和依赖；不启动录音，也不会显示 API key 内容。
- 配置 pytest 开发依赖、Phase 1 契约测试及 macOS/Ubuntu GitHub Actions。

### 验证

- `uv run pytest`：3 项通过。
- `uv run meetagent doctor`：正常运行；当前无 Gemini key 被作为云端功能缺失提示，不阻断本地诊断。
- `uv run python -m compileall -q src tests spikes`：通过。

### 尚待补足

- doctor 不主动打开麦克风或加载 Whisper 权重。真实采集权限与模型推理会在 Phase 2 的采集/ASR 验证中检查；Phase 1 验收通过。

## Voice interaction clarification — 2026-09-29

- 用户确认目标交互是“开会中 Gemini Live 持续听取会议、默认静默；用户直接问‘Agent，你怎么看？’时回答”，不需要先单独说唤醒词。
- 更新架构文档与 ADR-002/004、Phase D/Phase 4 要求和 spike 提示词。持续音频上传到 Gemini 的同意和成本须在会前说明；本地 Whisper 仍是正式逐字稿来源。
- 这一交互已接入 `meetagent demo`：麦克风帧并发供本地 VAD/Whisper 和持久 Gemini Live session 使用，Live 默认为安静，直接点名提问时播放语音回答；用户插话时本地 RMS 门限立即清队列，服务端 interrupted 再同步清除。待用户在 M5 上实测确认。
- 保存实时 Live 输入转写作为延迟兜底（如果本地 Whisper 没有匹配到，会写成 Speaker 发言；本地识别稍后到达时替换兜底条目）；Agent Live 回复和 Analyst 语音答案都写入同一 `.txt`/`.jsonl` 时间线，标记 `speaker=Agent`。

## Phase D draft — 2026-09-29

### 已完成

- 建立 `meetagent demo --out ...` 单一命令入口，支持麦克风与 WAV 文件模式。
- 接入 Silero VAD、Apple Silicon MLX Whisper、本地简体转换、三语关键词表。
- 接入 Gemini 3.8 Flash Google Search grounding 和 Gemini 3.8 Live 语音输出。
- 实现转写 JSONL 与会后 Markdown 纪要输出。
- 可阅读的 `.txt` 和结构化 `.jsonl` 转写文件在会议开始时创建，每段识别完成后立即追加并同步到磁盘；普通文件夹和 Obsidian vault 都保存逐字稿。
- 增加 `--language auto|zh|en|de`，将选定语言传给本地 Whisper 并记录在 JSONL。
- 将 Ctrl+C 改为受控停止录音后再等待 Gemini 总结；Gemini 未配置或请求失败时也写入本地 Markdown 纪要，终端明确说明来源。
- 实时麦克风音频保存为 WAV；离线 WAV 输入会复制到输出目录，音频、逐字稿和纪要使用同名文件前缀。
- 增加中文试用文档和 WAV 测试信号 fixture。

## Phase 2 local transcription — 2026-09-29

### 已完成实现

- 抽出可复用的 Silero VAD segmenter、本地 Whisper 后端和 TXT/JSONL transcript store；本地 whisper.cpp 模型在进程内复用。
- 新增仅本地运行的 `meetagent transcribe --file ... --out ...`，经 VAD 切段后输出音频、TXT、JSONL 和 Markdown。
- 接入 sherpa-onnx speaker embedding 匹配：未注册发言人按会议内 embedding 聚类；`meetagent enroll --name ... --file ...` 注册姓名和声纹向量。
- sherpa-onnx ONNX runtime 与 Python bindings 加入依赖；声纹模型首次运行下载并缓存，可通过 `MEETAGENT_SPEAKER_MODEL` 指定本地模型。

### 边界

- 按用户要求未运行真实音频/硬件验收；当前样本仍不是语音。
- 当前是按 VAD 段做 speaker attribution，不支持重叠语音源分离。姓名注册只保存 embedding，不保存注册 WAV。

## Phase 3 analyst — 2026-09-29

### 已完成实现

- 扩展三语触发表，增加显式逻辑检查请求。
- fact check 与逻辑检查使用 JSON schema 限制的结构化输出；核查依赖 Gemini Google Search grounding，逻辑问题必须保留发言原句和时间。
- 新增 OpenAlex、Semantic Scholar、arXiv scholarly search adapter；并发请求、索引失败时保留其余可用索引结果，按 DOI/标题去重。
- 研究方向只呈现模型选择且能与真实检索标题精确匹配的论文，Markdown 链接由检索记录提供，避免模型生成论文 URL。
- `doctor` 增加 sherpa-onnx runtime 和声纹模型缓存状态。
- 新增可复用 `GeminiAnalyst` 服务，将结构化 finding 发布到 EventBus 并保存到同名 `.findings.jsonl`，包含会议信息、类型、判断、置信度与来源。

### 约束

- 定时 claim extraction 通过显式 `--analyze-claims` 选项运行；默认关闭，启动时说明每分钟会额外向 Gemini Flash 发送最近两分钟转写，结果静默发布到 Finding 事件流。
- 按用户要求未运行实音、Gemini、学术索引或设备验收。

### 偏离计划之处

- 没有用户提供的自录音频，fixture 暂用合成测试信号，不能验证 ASR，也不能替代真实混说样本。
- Linux pywhispercpp 离线后端已接入，但要求用户提供 GGML 模型路径；Linux 实时麦克风仍未接入。
- 尚未在目标 Apple M5 硬件上验证延迟与 API 花费，因此 Phase D 验收未通过。
- 初版优先形成可试用链路，自动化测试未补齐。
- 当前运行容器无法访问 Metal，因此离线 ASR 未能在此处实测；CLI 会在桌面 Apple Silicon 环境执行 MLX 推理。

### 需要用户决定的问题

- 请提供或录制一段可公开在本机仓库使用的中英德混说 WAV 样本，以完成离线验证。
- 请在 Apple M5 上试用并反馈 VAD 分段、关键词触发、语音播放与纪要格式问题；根据结果再进入后续阶段。
