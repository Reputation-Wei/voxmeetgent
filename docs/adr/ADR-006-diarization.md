# ADR-006：说话人分离与注册

- 状态：已实现基础 embedding 识别与注册；多人重叠分离未实现
- 日期：2026-09-29

## 决策

采用 sherpa-onnx 的 speaker embedding extractor 与 manager。会前通过 `meetagent enroll` 从单人 WAV 提取 embedding 并登记姓名；会议中对每个 VAD 语音段计算 embedding，匹配已注册人员，未命中的 embedding 在当前会议内聚类为 `Speaker N`。模型由 sherpa-onnx 官方 release 下载到用户缓存，也可用 `MEETAGENT_SPEAKER_MODEL` 指定本地文件。

## 证据与限制

未用带标注的多人会议样本运行准确率基准。该方案为按 VAD 语音段分配声纹身份，不包含重叠语音的源分离；同一片段若有多人同时发言，不能恢复独立说话人轨道。Embedding 注册文件保存在用户本机 `~/.local/share/meetagent/speakers.json`，原始注册音频不保留。

## 后果

声纹 embedding 属于敏感数据，当前只保存在本机；删除注册项需从 `speakers.json` 移除对应记录。当前尚无 CLI 改名/删除命令。
