# 会议 Agent —— 给 Code Agent 的执行计划

> 目标读者：code agent（Claude Code / Codex 等）。
> 本计划按阶段执行，每个阶段结束都要停下来汇报结果和待决问题。用户于 2026-09-29 明确要求：参考 `cuppibla/live-dj` 加入实时语音反馈与可打断对话，并在 Phase D 尚未完成全部实机验收时继续 Phase 0；本轮按此要求推进。

---

## 0. 项目背景（code agent 必读）

一个本地运行的会议助手，工作名 `voxprogent`（可改）。

**功能需求**

1. 记录会议中每个人说的话（区分说话人），语言：中文 / 英文 / 德文，会有中英德混说。
2. 需要时 agent 参与讨论（语音回答）。
3. 对与会者的发言做 fact check；发现逻辑问题时指出。agent 需要搜索工具。
4. 关键词触发，例如「有没有相关研究」「is there related work」「gibt es Forschung dazu」。
5. agent 根据讨论内容给出排序后的研究方向建议（带理由和依据）。
6. 会后总结全部对话（纪要、决议、待办、研究方向）。

**技术约束**

- 语音识别：**本地 Whisper**。
- 语音对话：**Gemini Live API**。
- 目标硬件：
  - macOS，Apple M5（Metal）
  - Omarchy（Arch Linux + Hyprland），Intel Core Ultra X7 358H（Arc iGPU + NPU）
- 分发：macOS 通过 **Homebrew**（自建 tap）安装；Linux 做成 **Omarchy 可安装的扩展**。
- 用户 Python 基础有限：代码要清晰、注释充分、避免过度抽象的魔法写法。

---

## 1. 外部事实与 Phase 0 复核记录

2026-09-29 已复核 [Gemini Live API skill](https://github.com/google-gemini/gemini-skills/blob/main/skills/gemini-live-api-dev/SKILL.md)、[Live API best practices](https://ai.google.dev/gemini-api/docs/live-api/best-practices)、[session management](https://ai.google.dev/gemini-api/docs/live-api/session-management)、[tool use](https://ai.google.dev/gemini-api/docs/live-api/tools)、[thinking](https://ai.google.dev/gemini-api/docs/live-api/thinking) 和 [`live-dj`](https://github.com/cuppibla/live-dj) 的 `frontend/main.js`、`genai_sdk/raw_server.py`。下列 API 要点据此更新；实机行为仍由 Phase 0 spike 验证。

- **Gemini Live 当前推荐模型**：`gemini-3.8-live`（稳定版，默认低延迟语音 agent）和 `gemini-3.8-live-extended-thinking`（高推理，后台思考）。`gemini-3.1-flash-live-preview` 已是 legacy。
- `gemini-3.8-live` 支持 function calling 和 Google Search grounding；异步函数调用 `behavior: NON_BLOCKING` 为默认。extended-thinking 版**只接受** NON\_BLOCKING 工具，声明 BLOCKING 会报错。
- 会话限制：不开压缩时纯音频会话约 15 分钟；单个连接约 10 分钟，需要 **session resumption** 重连；上下文上限 128k token，音频约 25 token/秒。**必须开启 `contextWindowCompression`**。
- 计费：截至 2026-09-29，Gemini Live audio 输入标价为 $0.005/分钟，audio 输出 $0.018/分钟；输入与输出分钟分别按实际时长计费，另有 text token 与 Search grounding 费用。长会议持续上传麦克风会带来隐私暴露和持续输入成本。Gemini 3.8 Live 的 proactive audio 是模型默认行为，不能把它描述为可开关的模型配置；应用仍须通过指令和客户端 InterventionPolicy 控制用户听到的介入频率。见 ADR-002/004。
- Live API 输入 16 kHz 16-bit PCM，输出 24 kHz PCM。
- 实时输入使用 `send_realtime_input`，音频以 20–40 ms 小块传送且避免显著缓冲。服务端在用户打断时发送 `server_content.interrupted=true`；客户端必须清空播放队列。为获得更快的 barge-in，客户端可像 `live-dj` 一样用麦克风 RMS/VAD 在本地立即停止当前播放，再处理服务端 interrupted 事件。
- `session.receive()` 是按轮次结束的异步生成器，接收任务必须在外层循环中持续重启；麦克风上行与 Live 下行必须并发运行。
- `live-dj` 的前端采用 16 kHz 麦克风 PCM、24 kHz 输出播放、本地 RMS 门限打断和服务端 `interrupted` 兜底；raw SDK 服务端并发运行上行/下行任务并循环调用 `session.receive()`。这是交互实现参考，不直接照搬 DJ 音乐播放或浏览器架构。
- Google 另有 `gemini-3.5-transcribe(-live)`（云端，带说话人分离），**不作为默认**（用户要求本地 ASR），但可作为可选的对比/兜底后端。
- 官方的 Live API 开发 skill：`github.com/google-gemini/gemini-skills/skills/gemini-live-api-dev/SKILL.md` —— Phase 0 先读它和 ai.google.dev 的 Live API 文档（tools / session-management / best-practices / thinking 四页）。

---

## 2. 核心架构（草案；Phase 1 已落为架构文档与 ADR）

```
┌──────────────┐   PCM   ┌──────┐  segments  ┌──────────────┐   ┌──────────────┐
│ AudioCapture │ ──────▶ │ VAD  │ ─────────▶ │ Local ASR    │──▶│ Diarization  │
│ mic / 系统音频 │         │silero│            │ whisper      │   │ + 说话人注册  │
└──────────────┘         └──────┘            └──────────────┘   └──────┬───────┘
                                                                       ▼
                                                           ┌────────────────────┐
                                                           │ TranscriptStore    │
                                                           │ SQLite + JSONL     │
                                                           └─────────┬──────────┘
                                                                     │ EventBus (asyncio)
            ┌───────────────────────┬───────────────────────┬────────┴──────────────┐
            ▼                       ▼                       ▼                       ▼
   ┌────────────────┐    ┌──────────────────┐    ┌────────────────────┐   ┌──────────────┐
   │ TriggerDetector│    │ Analyst          │    │ InterventionPolicy │   │ Summarizer   │
   │ 关键词/唤醒词   │───▶│ gemini-3.8-flash │───▶│ 何时说话/只弹卡片   │   │ 会后纪要      │
   │ zh/en/de       │    │ + google_search  │    │                    │   └──────────────┘
   └────────────────┘    │ + 学术检索工具    │    └─────────┬──────────┘
                         └──────────────────┘              ▼
                                                  ┌────────────────────┐
                                                  │ VoiceAgent         │
                                                  │ Gemini Live        │──▶ 扬声器
                                                  │ gemini-3.8-live    │
                                                  └────────────────────┘
                     UI：跨平台本地 Web GUI（浏览器/PWA）+ Omarchy 状态栏插件；导出 Markdown（可写入 Obsidian vault）
```

### 关键设计决策（每条写成 `docs/adr/NNN-*.md`）

**ADR-001 本地 ASR 是"事实记录"的唯一来源**
Gemini Live 自带转写，但它不区分多人、且是云端。会议记录以本地 Whisper + 本地说话人分离为准；Live 的转写只用于 agent 自己的回合。

**ADR-002 Gemini Live 支持持续倾听、按请求发言与文本触发两种路径**

- **主交互路径：持续倾听、按请求回答**。开会期间麦克风以 16 kHz PCM 小块持续送入 Gemini Live 长连接，Live 持续听取会议上下文；默认保持安静，只有用户明确点名并提问时才回答，例如“Agent，你怎么看？”、“Hey Agent, what do you think?”、“Agent, was meinst du?”。本地 Whisper 仍是正式逐字稿唯一来源。用户可在 Agent 播报中直接插话，客户端 RMS/VAD 先停止本地播放，服务端 `interrupted` 信号清空剩余队列并同步模型状态。
- **文本触发路径**：不持续上传会议音频；需要 agent 回答时，发送近期本地转写与 Analyst 结论，再由 Live 语音回答，作为隐私/成本优先选项。
- 默认介入仍为 `on_request`，即“持续听、未被点名不主动说话”。实时语音路径会把整场会议麦克风流发送到 Google；开始时必须明确提示并取得必要同意。客户端 policy 控制实际播放和主动发言频率。
- Phase 0 分别测两条路径的首音频延迟、可打断性和费用，并检查长会话恢复；实时语音是用户指定的主要交互路径，实测用于确定可行配置与每小时成本。

**ADR-003 实时对话和深度分析分开**
Live 模型负责"说"，要快；fact check、逻辑检查、研究方向需要检索和推理，放到后台 `Analyst`（`gemini-3.8-flash` 普通 API + `google_search` grounding + 自定义学术检索工具），结果再交给 VoiceAgent 播报或只显示在 UI 上。Live 侧所有工具声明为 `NON_BLOCKING`，避免检索时冷场。

**ADR-004 介入策略（InterventionPolicy）必须可配置**
会议 Agent 默认持续接收实时语音，但不等于主动发言。默认 `on_request` 只有在检测到对 Agent 的明确称呼/提问后才回答。模式：

- `silent`：只在 UI 上弹卡片（fact check 结果、逻辑问题）
- `on_request`：只有唤醒词/关键词才开口（默认）
- `proactive`：高置信度问题主动发言，但有频率上限（如每 5 分钟最多 1 次）和置信度阈值
指出逻辑问题时语气要中性、给依据和来源；不确定时说"可能"，不下结论。

**ADR-005 ASR 后端抽象**
接口 `ASRBackend.transcribe(segment) -> list[Utterance]`，实现：

- `whispercpp`：调用 whisper.cpp 的 `whisper-server`（macOS 用 Metal；Linux 用 Vulkan，评估 OpenVINO 用于 NPU）—— **默认**，因为 brew 和 Arch 都有现成包，打包最简单
- `mlx`：`mlx-whisper`，仅 macOS，Phase 0 对比速度
- `gemini_transcribe`：可选云端兜底
当前 macOS Apple Silicon 默认模型为完整 `large-v3`（MLX 转换版 `mlx-community/whisper-large-v3-mlx`），优先识别质量；M5 首次使用自动下载约 3.1 GB 权重。可用 `MEETAGENT_WHISPER_MODEL_ID=mlx-community/whisper-large-v3-turbo` 切回低延迟选项。中文输出统一转简体（OpenCC）；每段单独做语言检测以支持中英德混说。Linux 模型仍由 `MEETAGENT_WHISPER_MODEL` 指向本地 GGML 文件。

**ADR-006 说话人分离**
默认 `sherpa-onnx`（ONNX，无 torch 依赖，便于 brew/Arch 打包）；`pyannote` 作为可选高精度后端（需要 torch + HF token）。支持"说话人注册"：会前每人说一句，存 embedding，会中把 `SPEAKER_0` 映射成真实姓名；UI 可手动改名并回写。

**ADR-007 回声问题**
agent 说话会被麦克风录进去，也可能触发客户端 barge-in。macOS 用系统 AEC/voice processing；Linux 用 PipeWire `echo-cancel`。结合 AEC、播放器输出参考和 RMS/VAD 门限抑制扬声器回灌；本地 Whisper 对重叠片段标记 `speaker=agent` 并排除出 fact check。必须用扬声器和耳机分别验证误触发率。

**ADR-008 音频来源**

- 线下会议：麦克风。
- 线上会议：系统音频回环（macOS：ScreenCaptureKit 或 BlackHole；Linux：PipeWire monitor source）+ 麦克风混合。

**ADR-009 隐私与同意**
会议开始时 UI 必须分别提示本地录音/转写，以及实时语音模式会将麦克风流发送到 Gemini；须取得与会者必要同意。当前 Demo 按用户要求保存本地 WAV。正式版的音频保存默认值在 Phase 1 ADR 中确认，并提供文本触发路径以避免持续上传。

### 实测记录（2026-09-29，Apple M5）

- 硬件确认：Apple M5，Metal 4；当前 shell 没有 `GEMINI_API_KEY` 或 `GOOGLE_API_KEY`，Live 网络调用 spike 尚未执行。
- `mlx-whisper` 在 M5 上成功加载 `large-v3-turbo`。系统 `say` 合成的三个独立短句固定语言测试，各运行 3 次并取常驻进程中位数：中文 5.88 s / 0.430 s（RTF 0.073），英语 6.39 s / 0.364 s（RTF 0.057），德语 4.84 s / 0.392 s（RTF 0.081）。whisper.cpp `whisper-server` 使用 Metal 和 GGML large-v3-turbo-q5_0（547 MB），相同样本中位数分别为 0.540 s（RTF 0.092）、0.502 s（RTF 0.079）、0.519 s（RTF 0.107）。两后端在固定语言样本上都逐字正确；模型格式和量化不同，这不是严格精度对比。MLX 新进程首轮约 1.17–1.23 s，随后约 0.36–0.43 s。
- 自动语言检测没有保留合成混说片段中的全部语言：整段测试主要返回德语，Demo 的 8 秒窗口识别出中文和德语、漏掉英语。固定语言仅在对应单语样本验证成功。合成音频不能代表真实会议；无自录三语样本，故不报告真实会议 WER/CER。
- 用户本机已成功运行文本触发语音回复 spike：连接 0.444 s、请求至首段音频 0.837 s、收到音频总长 10.220 s，并写入 `/tmp/live-text-response.wav`。脚本保存文件但不播放，因此不算扬声器播放或打断实测。
- 用户本机初次耳机打断测试：24.592 秒，10 次本地 RMS 停播、2 次服务端 `interrupted`、0 丢帧；发现连续语音重复计数和异常语言识别。
- 调整去重锁存后复测：16.889 秒，2 次讲话对应 2 次本地 RMS 停播与 2 次服务端 `interrupted`，0 丢帧。第一句英语 prompt 被正确识别，后续插话只识别为 “Aten”。确认了耳机下基础打断链路能停播并收到服务端中断，但多语识别不可靠、扬声器回声行为和定量停播延迟仍未测。monotonic 时间戳不能作为延迟测量。
- 持续音频、20 分钟会话恢复与 NON_BLOCKING 工具 spike 尚未在用户设备执行。两个麦克风 spike 现在要求显式 `--consent-to-stream` 后才开麦克风。
- diarization benchmark 未实现/运行：缺少 3–4 人带标注样本和 pyannote token；Linux Vulkan/OpenVINO 也需目标 Linux 设备。
- API 费率来自[官方价格表](https://ai.google.dev/gemini-api/docs/pricing)：连续 1 小时 Live 音频输入基线约 $0.30；每 1 分钟模型语音输出另约 $0.018。推理文本、搜索 grounding 和其他 Analyst 请求另计。实时模式 10 分钟的最低输入音频基线约 $0.05，语音输出按实际时长再计；不是 API 实测账单。

---

## 3. 目录结构（Phase 1 生成骨架）

```
meetagent/
├── pyproject.toml            # uv 管理，Python 3.12
├── src/meetagent/
│   ├── cli.py                # `meetagent start|stop|summary|enroll|doctor`
│   ├── config.py             # pydantic-settings，读 ~/.config/meetagent/config.toml
│   ├── secrets.py            # API key：macOS Keychain / Linux libsecret，兜底环境变量
│   ├── events.py             # EventBus + 事件数据类（Utterance, Trigger, Finding …）
│   ├── audio/                # capture.py, vad.py, aec.py
│   ├── asr/                  # base.py, whispercpp.py, mlx.py, gemini_transcribe.py
│   ├── diarization/          # base.py, sherpa.py, pyannote.py, enrollment.py
│   ├── store/                # transcript.py（SQLite + JSONL）
│   ├── triggers/             # keywords.py（三语关键词表 YAML）
│   ├── analyst/              # factcheck.py, logic.py, research.py, prompts/
│   ├── tools/                # scholar.py（OpenAlex / Semantic Scholar / arXiv）
│   ├── voice/                # live_session.py（连接、压缩、resumption、重连）
│   ├── policy.py             # InterventionPolicy
│   ├── summary/              # summarizer.py, templates/（中/英/德纪要模板）
│   └── ui/                   # 本地控制 API、WebSocket 事件流与前端资源
├── packaging/
│   ├── homebrew/meetagent.rb
│   └── omarchy/              # PKGBUILD, systemd user unit
├── omarchy-plugin/           # Quickshell 插件：manifest.json, BarWidget.qml, Panel.qml
├── spikes/                   # Phase 0 验证脚本
├── tests/fixtures/           # 三语混说测试音频（自己录，不用受版权保护素材）
└── docs/ARCHITECTURE.md, docs/adr/
```

---

## 4. 分阶段执行（每阶段结束停下来汇报）

### ⭐ Phase D —— 最小 Demo（**最先做**，完成后停下来给我试用）

目标：一天内能在 Mac 上跑起来，亲眼看到"持续倾听 + 默认静默 + 点名后实时语音回答 + 可打断 + 本地转写 + 会后纪要"整条链路。**不追求架构完美，但代码放在第 3 节的目录里，后续阶段在它上面长。**

范围（只做这些）：

1. **单一入口** `meetagent demo --out <文件夹> --consent-to-stream`。实时麦克风模式必须显式确认整场音频会发送到 Gemini Live；没传 `--out` 或没确认同意时拒绝启动。
2. **音频**：只用默认麦克风，`sounddevice` 采集 16 kHz 单声道。
3. **VAD + ASR**：silero-vad 切段 → 本地 whisper（macOS 用 `mlx-whisper`，Linux 用 `pywhispercpp` 或 whisper.cpp server，哪个装得快用哪个），macOS 默认精度优先 `large-v3`、可选 `large-v3-turbo`，中文转简体。
4. **不做说话人分离**：统一标为 `Speaker`。（Phase 2 再加）
5. **终端输出**：用 `rich` 实时打印「\[mm:ss\] 文本 (语言)」即可，不做 Textual TUI。
6. **关键词触发**：三语关键词 YAML，例如：
   - 唤醒词：「agent」「助手」「Hey Agent」；例如直接问“Agent，你怎么看？”即可触发，无需先单独唤醒再问。
   - 研究：「有没有相关研究」「related work」「gibt es Forschung」
   - 核查：「这是真的吗」「fact check」「stimmt das」
7. **触发后**：
   - 把最近 2 分钟转写 + 触发类型发给 `gemini-3.8-flash`（开 `google_search`），拿到带来源的文本答案，终端打印答案和来源链接。
   - 持久 Gemini Live（`gemini-3.8-live`）会在会议期间持续听取 20–40 ms 的 16 kHz PCM 块；默认静默。用户直接问“Agent，你怎么看？”或“Hey Agent, what do you think?”即用语音回答，不要求额外按键或先发独立唤醒词。Local Whisper 仍独立负责本地逐字稿。研究/事实核查问题由本地转写触发 Analyst 搜索，再交给主播放队列播报，避免与 Live 重复发言。
   - agent 播报期间允许与会者直接插话：本地 RMS/VAD 门限触发后立即停止并清空待播音频；同时继续向 Live 发送用户语音。处理服务端 `interrupted=true`，清除剩余播放队列。实时上行和 Live 接收必须并发，持续循环 `session.receive()`。
   - 默认介入策略维持 `on_request`（一直听，只有被明确点名/提问才说）；实时模式启动时明确提示整场麦克风音频会发给 Gemini。文本触发模式仍可作为减少上传的选项。
8. **结束**：Ctrl+C → 用 `gemini-3.8-flash` 生成 Markdown 纪要（摘要、要点、agent 回答过的问题及来源），写入 `--out` 目录；若目录是 Obsidian vault 则写入 `Meetings/` 并加 frontmatter。同时保存原始转写 `.jsonl`。
9. **API key**：从环境变量 `GEMINI_API_KEY` 读取（Keychain 放到正式版）。
10. **开始时**终端显示一行"本会议正在转写"的提示。

明确**不做**：说话人分离、系统音频、独立学术检索工具、TUI、打包、Omarchy 插件。语音只在用户明确点名提问或提出搜索/核查请求时回答；高频 proactive 分析与介入留在后续阶段。

交付物：

- `README-demo.md`（中文）：安装步骤（`uv sync`）、如何设 key、如何运行、已知限制。
- `tests/fixtures/` 里一段自录的三语样例 + `meetagent demo --file <wav>` 离线模式，方便不开麦克风也能测。

**验收**：

- 启动命令明确同意实时上传后，Agent 整场倾听但未被点名时保持静默；直接问“Agent，你怎么看？”后用语音回答，无需独立唤醒轮次。在 Mac 上说一段中德英混合的话，终端 5 秒内出现本地转写。
- 说"有没有相关研究……"，10 秒内终端出现带链接的答案，并听到 agent 朗读；使用实时语音路径时可自然插话，插话后本地播放应立即停止，服务端 interrupted 后不得继续播旧音频。
- Ctrl+C 后指定文件夹里出现纪要 `.md` 和转写 `.jsonl`。
- 同一会议输出目录包含原始 WAV、可读 `.txt` 和结构化 `.jsonl`；本地 ASR 的与会者发言、Live 输入转写兜底和 Agent 回复共享时间线，Agent 明确标记为 `speaker=Agent`。
- 报告：实测延迟、一次 10 分钟 demo 的 API 花费估算、遇到的问题。

Demo 验收通过后，再按下面的 Phase 0→6 推进（Phase 0 中 Demo 已覆盖的 spike 可以跳过，只补缺的）。

### Phase 0 —— 技术验证（spike），不写正式代码

1. 读取 Gemini Live 官方文档和 gemini-skills 的 `gemini-live-api-dev/SKILL.md`，核实模型、流式收发、音频格式、打断、长连接、压缩、resumption、工具行为；发现偏差就更新本文件。
2. `spikes/live_text_in_audio_out.py`：文本模式，发送带最近会议上下文的文本，Live 用语音回答；测连接时间和首个音频块延迟。
3. `spikes/live_audio_passthrough.py`：实时路径，推麦克风音频，测端到端反馈延迟、云端识别/回复延迟、会话重连；配置 `contextWindowCompression` 和 session resumption，跑 20 分钟以上验证上下文恢复。按键未配置或没有与会者同意时不推流。
4. `spikes/live_barge_in.py`：复刻 `live-dj` 的本地 RMS/VAD 即时停止播放 + Live 服务端 `interrupted` 清队列；比较耳机/扬声器、RMS 门限和端到端停止延迟，记录自回声误触发。
5. `spikes/asr_bench.py`：whisper.cpp vs mlx-whisper，在 M5 上跑经用户许可的三语混说样本，记录实时率（RTF）、延迟和错误率；Linux 上测 whisper.cpp Vulkan，评估 OpenVINO/NPU 是否值得。
6. `spikes/diarization_bench.py`：sherpa-onnx vs pyannote，3–4 人样本的准确率和速度。
7. `spikes/nonblocking_tool.py`：Live 中调用一个耗时 5 秒的 NON\_BLOCKING 工具，确认语音反馈与打断不中断。
8. 输出 `docs/spike-report.md`：分别报告文本/实时音频路径的延迟、每小时成本估算、barge-in 实测、ASR/分离数据、限制和推荐选择。

**验收**：spike-report 有实测数字；列出需要我拍板的决策。

### Phase 1 —— 架构文档 + 骨架（已完成，2026-09-29）

1. 根据现有 spike 结果写 `docs/ARCHITECTURE.md` 和 ADR-001～009；未经实测的选择标为暂定。
2. 生成第 3 节主要目录结构，模块只有接口、数据类、docstring 和 `NotImplementedError`；`events.py` 和 `config.py` 实现配置/事件契约。
3. `meetagent doctor`：检查本机音频依赖、whisper 模型/后端、API key 是否存在（不显示值）、输出路径。
4. 单元测试框架（pytest）与 CI（GitHub Actions：macOS + Ubuntu）。

**验收**：`uv run meetagent doctor` 可运行；`pytest` 通过；我能读懂每个模块职责。目录结构以实际代码为准，保持本项目 `src/meetagent/` 包名。

**2026-09-29 执行记录**：`docs/ARCHITECTURE.md`、ADR-001～009、`config.py`、`events.py`、主要模块接口、`meetagent doctor`、pytest 开发依赖、3 个 Phase 1 契约测试和 macOS/Ubuntu CI 已建立。`uv run pytest` 3 项通过；`uv run meetagent doctor` 可运行，并检查输出目录写权限、系统输入设备列表、Metal 可用性和依赖。doctor 不会打开麦克风或实际加载 Whisper 权重；授权和真实推理留到对应功能阶段验证。Phase 1 验收项通过。

### Phase 2 —— 转写流水线（纯本地，不接 Gemini）

实现：离线 `meetagent transcribe --file <wav> --out <目录>` 使用 Silero VAD 切分语音、MLX Whisper（Apple Silicon）/ whisper.cpp（Linux）本地识别，并输出 `.wav`、`.txt`、`.jsonl`、`.md`。麦克风 Demo 继续使用相同 VAD/ASR 与 transcript Store；终端按「[时间] 姓名: 内容」滚动显示。Linux whisper.cpp 模型在进程内复用。

说话人识别使用 sherpa-onnx embedding extractor：`meetagent enroll --name <姓名> --file <单人 WAV>` 注册姓名，保存声纹向量到 `~/.local/share/meetagent/speakers.json`；未注册说话人经多片段确认后才按 embedding 聚类为 `Speaker N`，否则标记“说话人未知”。首次使用会把官方 ONNX embedding 模型下载到 `~/.cache/meetagent/models/`。当前逐段识别能为轮流发言提供候选归属，不能把同一时间的重叠语音分离成多路音频。终端采用 Rich 文字滚动输出；图形界面规划于 Phase 6。

**阶段状态（2026-09-29，更新）：** Phase 2 基本实现。根据用户对多人会议逐字稿保全与识别质量的要求，macOS 默认 Whisper 从 `large-v3-turbo` 改为完整 `large-v3`；麦克风回调保存原始单声道 WAV，并约每秒 fsync，转写 TXT/JSONL 在每个已识别发言后立即写入并 fsync。ASR 处理中采集帧使用无界本地队列，避免原 4 秒队列满后静默丢帧；若 VAD/ASR 失败仍保留 WAV 并报告错误。Live 点名门控依据本地 Whisper 识别文本，向 Live 显式发送该文本后才授权本回合语音输出。2026-09-29 后续调整：未注册声纹需满足时长/音量门限并经多片段相似度确认才分配标签；新增 `speaker-correct` 同步修订 TXT/JSONL、声纹重复注册融合；Whisper 增加近静音/短片段门限和解码窗级低置信过滤，明显重复文本拦截，相关阈值可由环境变量配置。**识别结果无法保证逐字无误**：重叠讲话、远场麦克风、噪声仍会造成错误；当前单声道逐段说话人识别也无法分离同时发言。上述规则尚无用户真实会议样本量化验证，需注意筛选可能漏掉轻声或短词；真实 WER/CER、长会持续录音和 Gemini 点名链路仍待实测。

### Phase 3 —— Analyst（fact check / 逻辑 / 研究）

当前实现：事实核查与逻辑检查由对应三语触发词显式启动；最近两分钟逐字稿作为分析窗口，模型返回 schema 约束的 JSON，再呈现 Claim/Verdict/Confidence/Evidence 或带时间戳原话的逻辑问题。研究触发并行查询 OpenAlex、Semantic Scholar、arXiv；方向中的论文只保留与检索返回标题精确匹配的记录，最终链接来自真实检索记录。普通回答、核查和研究均仅在触发后播报，遵守会议中的默认静默要求。

`GeminiAnalyst` 现在作为可复用服务连接结构化分析、`Finding` EventBus 和 `meeting-时间戳.findings.jsonl` 持久化。每条 finding 带 meeting id、事件时间、类型、结论/状态、置信度（如适用）和来源链接。

周期性后台 claim extraction 已作为显式选项 `--analyze-claims` 接入麦克风模式：每分钟使用最近两分钟逐字稿抽取可核查陈述、发布 Finding 并写入事件文件；默认关闭，启动时提示额外 transcript 上传到 Gemini Flash，不会播报。

**阶段状态（2026-09-29）：显式触发的核查、逻辑分析、scholar 检索、`Finding` 事件流和可选后台 claim extraction 已实现；按用户要求未做真实 Gemini/外部索引验收。**

1. 滚动窗口抽取"可验证的事实陈述"（claim extraction），结构化输出。
2. fact check：`gemini-3.8-flash` + `google_search`，输出 {claim, verdict, confidence, sources}。
3. 逻辑检查：前后矛盾、偷换概念、以偏概全、因果倒置等，必须引用原话和时间戳。
4. 研究检索工具 `tools/scholar.py`：OpenAlex / Semantic Scholar / arXiv，返回真实论文（标题、年份、DOI/链接），**禁止模型编造文献**——只能引用工具返回的结果。
5. 研究方向：基于讨论 + 检索结果，输出候选方向列表，每条含：问题、新颖性依据（相关工作）、可行性、风险、排序理由。
6. 三语关键词触发表（YAML，可自定义）。

**验收**：fixture 中故意放入的错误事实和逻辑矛盾能被发现；研究推荐里的每篇论文链接都真实可访问。

### Phase 4 —— VoiceAgent（Gemini Live）+ 介入策略

Live 会话管理（压缩、resumption、断线重连、goAway 处理）；InterventionPolicy 三种模式；回声消除；agent 用触发者的语言回答。

**验收**：1 小时模拟会议不断线；agent 说话不会被记成与会者发言；`on_request` 模式下不会乱插话。

**实现进度（2026-09-29，本轮继续）**：三种策略已接入配置文件和 `--intervention` CLI 覆盖；所有非静默模式均只接受明确的“Agent，你怎么看？”或英/德等价呼叫，Live 输入分片在短窗口内合并判断，且本地 Whisper 识别完成后也能补发同一回合授权；提及 Agent 和普通问句不开放播放。Live 回合播放授权不会串到后续回合；连接错误使用 1、2、4、8、16、30 秒封顶退避，并保留 session resumption。近期 Agent 输出相似文本会从本地/Live 输入转写中过滤，Agent 自身回答仍记为 `speaker=Agent`。可选 `pywebrtc-audio` extra 已接入声学 AEC：扬声器回调生成 16 kHz far-end 参考，AEC 清理麦克风信号后再供 RMS 插话检测、本地 Whisper 和 Gemini Live 使用；WAV 仍保存原始麦克风输入。README 已补充依赖、配置和使用说明。

**状态：实现代码已覆盖 Phase 4 范围，验收尚未完成。** 用户进一步明确所有策略下都必须先检测到明确呼叫“Agent，你怎么看？”（或英/德对应句）才授权播报；Proactive 不能越过客户端门控。M5 本机 MacBook Air 麦克风/扬声器以低电平带限噪声做了 12 秒 AEC 参考检查，没有保存音频：比较 0/20/35/50/75/100 ms 时，20–35 ms 在大多数播放窗口比 50 ms 抑制更强；单独 20 ms 复测的六个播放窗口，清理后 RMS 相对原始分别下降约 5.9–14.0 dB。该指标包括房间噪声、没有用户语音，不能证明对真实语音的质量，因此默认改为 20 ms。当前执行环境未检测到 `GEMINI_API_KEY`，所以尚未验证 Live 是否仅在呼叫后回答；一小时重连、真实插话、语音 AEC 和三语会议验收也未完成。

### Phase 5 —— 会后总结

`gemini-3.8-flash` 长上下文处理全文 → Markdown 纪要：摘要、按议题分段、决议、待办（负责人/截止）、fact check 记录、研究方向。可选写入 Obsidian vault 指定文件夹（路径可配置）。纪要语言可选 zh/en/de。

**阶段状态（2026-09-29）：实现已完成，Gemini 端到端验收待用户本机运行。** 已实现长逐字稿按发言边界分块提要、最终合并；zh/en/de 纪要语言；议题、决议、待办、事实核查与研究 Finding、未解决问题、Agent 回答及来源结构；缺少 key/接口失败时输出对应语言的本地逐字稿型 Markdown。Markdown 先原子保存至 `--out`，可通过 `--obsidian-vault`/`--obsidian-folder` 或 `[output]` 配置额外写入 Obsidian；同步前检查 `.obsidian` 和路径越界，失败时保留本地纪要。API key 不进文件。离线结构/存储验收通过；当前 agent 运行环境无 Gemini key，模型纪要质量、超长会议与用户真实 vault 验收未完成。

### Phase 6 —— 跨平台 GUI（macOS M5 + Omarchy）

**目标**：不用终端即可开始/结束会议、查看实时逐字稿和 Agent 活动、修正说话人，并打开音频与会后纪要。GUI 是现有 Python meeting service 的本地客户端，不在浏览器中重复实现音频、ASR 或 Gemini 管道。

**技术方向**：本地 Web GUI，由 Python 后端在 `127.0.0.1` 提供 API 与 WebSocket，前端用轻量静态网页实现；同一 UI 可在 macOS 与 Omarchy 桌面浏览器使用，也可安装为 PWA。Omarchy Quickshell 插件只作为可选状态栏入口，通过本地接口显示状态和打开面板，不复制完整 GUI。桌面原生打包以后再评估 Tauri；本阶段不以原生安装包为前提。

**后端边界与接口**：

- 将会议 start/stop/status 从 CLI 调用路径抽成可管理的异步 MeetingService；同一时间只允许一个活动会议。API 不得执行任意 shell 命令。
- `GET /api/status` 返回空闲/准备中/录制/停止中/错误、时长、输出路径、Live 连接与 Agent 播报状态；不得返回 API key。
- `POST /api/meetings` 校验输出目录、语言、纪要语言、Live 模式及同意选项后启动；`POST /api/meetings/{id}/stop` 正常收尾并 flush 音频、转写和纪要。
- `GET /api/meetings/{id}` 和分页 transcript endpoint 读取记录；speaker correction endpoint 同步更新 TXT/JSONL；summary endpoint 返回生成状态与文件位置。
- `WS /api/events` 推送 transcript append/correction、speaker change、Agent answer、finding、summary ready、连接状态和错误；事件带稳定序号，断线后能按序号补拉，不能静默漏事件。
- 默认仅绑定 loopback；检查 Origin/Host，以每次启动随机本地 token 或等效 CSRF 防护保护控制接口；不暴露到局域网。
- 本地录音/落盘、整场音频 Live 上传、会后逐字稿 Gemini 上传分别显示和授权。Live 必须提示整场音频会上传并取得用户确认；会后文字上传也单独 opt-in，默认关闭；停止按钮始终可用，窗口关闭不能无提示丢弃会议。

**页面和交互**：

1. **首页/会议控制**：开始、停止、输出位置、输入设备、主要语言、纪要语言、本地模式或实时 Live；依赖/权限缺失时展示具体诊断。
2. **实时会议页**：录音计时与保存状态、滚动逐字稿（时间/说话人/原始识别文字）、Agent 静默/播报状态、Live 连接和错误状态。
3. **说话人校订**：逐条改名、合并标签、标记未知，写回 TXT/JSONL；修订前显示原标签并可撤销最近一次修改。
4. **会后纪要页**：议题、决议、待办、Fact check、研究方向、未解决问题和 Agent 回答/来源；重新生成纪要、选择 zh/en/de、打开 WAV/TXT/JSONL 与 Obsidian 位置。
5. **隐私与错误**：区分本地录音保存和 Gemini 上传；清楚展示麦克风、模型下载、磁盘写入、API 断线、低置信转写问题，不把“已连接”显示成“已保存”。
6. **可访问性**：键盘可操作、焦点可见、状态变化使用文本标签而非仅颜色；中英德覆盖授权、录音和错误主流程。

**交付顺序**：先实现生命周期管理/API/WebSocket 协议及本地安全检查；再实现浏览器 UI，完成空闲/同意/录制/断线/结束闭环；随后增加逐字稿、说话人校订和纪要页。先在 M5 验证，再在 Omarchy 浏览器验证；Quickshell 插件仅做状态、开始/停止、打开面板入口。

**验收**：浏览器刷新或短暂断线后恢复状态并补齐逐字稿；未经同意不能开启 Live；每条转写及时显示；改 speaker 后 TXT/JSONL 一致且可撤销；结束后查看纪要、来源并打开原音频；M5 与 Omarchy 都可用；API 不能被局域网其他设备访问；麦克风/磁盘/Gemini 出错时 UI 清楚区分并保留已保存文件。

**实现进度（2026-09-29）：** 第一版已落地 FastAPI/Uvicorn 本机服务与浏览器界面：`meetagent gui` 启动服务；API 限制 loopback Host/Origin、使用进程随机 token；会议生命周期由 `MeetingService` 管理，支持本地/Live 显式同意模式、会后文字上传单独 opt-in（默认本地纪要）、浏览器停止、状态/逐字稿 WebSocket 增量与变更重置、TXT/JSONL/WAV/Markdown 打开、说话人修正与最近一次修正撤销。UI 为轻量静态网页，可直接在 M5 与 Omarchy 浏览器使用。仍待继续：稳定事件游标/断线历史补拉、speaker merge 与完整撤销栈、音频设备选择、Agent/Live 实时连接状态精确推送、UI 中英德翻译、真实浏览器/M5/Omarchy 安全与生命周期验收、PWA。Homebrew/AUR 安装器留给 Phase 7。

### Phase 7 —— 打包分发

**macOS / Homebrew**

- 自建 tap：`<github用户名>/homebrew-tap`，formula `meetagent.rb`。
- 依赖 `python@3.12`、`whisper-cpp`、`ffmpeg`；Python 依赖用 `Language::Python::Virtualenv`（用 `brew update-python-resources` 生成 resource 列表）。重依赖（torch/pyannote）**不进 formula**，做成可选 extra。
- `brew services` 支持后台运行；首次运行 `meetagent setup` 下载 whisper 模型、写 API key 到 Keychain、申请麦克风/屏幕录制权限。
- 验收：干净 macOS 上 `brew install <tap>/meetagent && meetagent doctor` 通过。

**Omarchy（Omarchy 4 "quattro"，基于 Quickshell 的插件）**

**实现进度（2026-09-29）：** 新增首版 Quickshell bar-widget 插件 `omarchy-plugin/`：读取 MeetAgent 本机 GUI 状态、显示录音计时、可开始本地录音/Live 会议、停止收尾并打开完整 GUI；音频流上传和纪要文本上传分别要求用户勾选同意。插件用随机 token 调用 loopback API，不保存 API key。当前只完成文件与接口静态核对；运行环境不是 Omarchy，Quickshell/QML 加载、`omarchy plugin validate`、麦克风权限和端到端录音尚未验证。插件暂要求用户先运行 `meetagent gui --no-open`（开发环境可用 `uv run meetagent gui --no-open`），后续需补自动启动/服务集成、转写面板、发行安装器并在 Omarchy 真机验证。详见 `omarchy-plugin/README.md`。

背景（已核实，code agent 仍需对照当前 omarchy 仓库复核）：

- Omarchy 4 用 Quickshell 实现的 `omarchy-shell` 取代了 Waybar + Mako；bar、面板、通知、菜单都以插件形式运行在这个长期运行的 Quickshell 进程里。
- 插件目录 `~/.config/omarchy/plugins/<plugin-id>/`，包含 `manifest.json`（schema 1）和 QML 文件。
- 安装/管理：`omarchy plugin add <git-url> --enable`、`omarchy plugin update|remove <id>`、`omarchy plugin validate`；手动安装后 `omarchy-shell shell rescanPlugins`。
- 插件设置保存在 `~/.config/omarchy/shell.json`（明文）—— **API key 不要放这里**，用 libsecret。
- CLI 与运行中 shell 的通信走 IPC：`omarchy-shell <plugin-id> <command>`。
- 参考实现：`omacom/omarchy` 仓库 `shell/plugins/bar/README.md` 和 `docs/omarchy-shell.md`；社区插件 `basecamp/omarchy-basecamp-plugin`、`quicknode/quicknode-omarchy-plugin`、`forbidden-game/omarchy-plugins`（里面有一个 ASR 插件，可参考）。

架构：**Python 后端 + QML 前端分离**

- 后端：`meetagent` 作为 systemd user service 常驻，提供本地控制接口（Unix socket 或 `127.0.0.1` HTTP/WebSocket），推送状态和事件（录制中/触发/fact check 结果/纪要完成）。macOS 和 Linux 共用这套后端。
- 前端：Quickshell 插件 `io.github.<用户名>.meetagent`：
  - Bar 组件：录制状态指示、计时、点击开始/停止。
  - 弹出面板（Panel.qml）：实时转写滚动、当前 fact check 卡片、介入模式切换、输出文件夹显示。
  - 通知：agent 结论和纪要完成用 omarchy-shell 的通知展示。
  - IPC 命令：`omarchy-shell <id> toggle|start|stop|panel`，方便绑定 Hyprland 快捷键。
- Arch 包：`PKGBUILD`（AUR）装后端 + whisper.cpp（Vulkan 版，Arc iGPU）；插件本身走 `omarchy plugin add`。
- `omarchy plugin remove` 不会清理 `shell.json` 中的条目 —— 卸载说明里要写明，或提供清理命令。
- 验收：`omarchy plugin validate` 通过；在 Omarchy 4 上一条命令装好，bar 上能开始/结束会议，面板里看到实时转写。完整跨平台 Web GUI 验收属于 Phase 6。

---

## 5. 通用要求

- 全部 `asyncio`；每个模块可单独测试；外部 API 调用都有 mock。
- 日志用 `structlog`，默认不记录音频和 API key。
- 配置示例 `config.example.toml`，注释写清楚每一项。
- 代码注释用英文，面向用户的文档和 README 提供中文版。
- 不确定的 API 行为：先查官方文档，查不到就写 spike 验证，不要凭记忆写。
- 每个 Phase 结束：更新 `docs/CHANGELOG.md`，列出"已完成 / 偏离计划之处 / 需要用户决定的问题"。

## 6. 用户已确认的决定

1. **音频来源：线下（麦克风）和线上（系统音频回环）都要支持。** Demo 只做麦克风，系统音频放到正式版。
2. **默认介入模式：`on_request`。**
3. **输出位置：** 提供"写入 Obsidian vault"选项；没有 vault 就写入用户指定的普通文件夹。
   - GUI/setup 将配置输出目录；当前配置字段为 `[output].directory`、`[output].obsidian_vault`、`[output].obsidian_folder`（默认 `Meetings`），环境变量 `MEETAGENT_OUTPUT_DIR`/`MEETAGENT_OBSIDIAN_VAULT` 可覆盖。避免同时保留另一套 `output.mode/path/subfolder` 命名。
   - 每次会议可临时覆盖输出目录；GUI 提供目录选择器。没有有效输出目录时不允许开始录音。
   - Obsidian 模式检查路径下存在 `.obsidian/`；纪要带 YAML frontmatter（date、participants、tags）。音频、TXT 和 JSONL 保存在主输出目录，Markdown 纪要可另外同步到 vault。
4. **Gemini 接入：先用 AI Studio API key**（google-genai SDK 的 Gemini Developer API 模式）。client 创建集中在一处，将来切 Vertex AI 只改这里。
