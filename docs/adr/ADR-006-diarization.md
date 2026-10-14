# ADR-006：说话人分离与注册

- 状态：已实现声纹识别、在线 EMA 质心自适应、候选查重合并与会后全局重聚类
- 日期：2026-10-14（更新）

## 决策

采用 sherpa-onnx 的 speaker embedding extractor 与 manager。会前通过 `meetagent enroll` 从单人 WAV 提取 embedding 并登记姓名；会议中对每个有效语音段（$\ge 0.8$ 秒，RMS $\ge 0.003$）计算 embedding。在线识别阶段引入指数滑动平均（EMA，$\alpha=0.85$）自适应跟踪说话人音色漂移，并在确认新候选前执行跨簇合并查重（相似度 $\ge 0.58$ 时并入已有发言人），彻底避免发言人碎片化膨胀。

会议结束时执行全局二次重聚类（`recluster_session`）：合并相似发言人、基于记录的特征向量与上下文连续性补全短句未标注发言，并规范化序号。同时支持 CLI 全场批量改名（`meetagent speaker-correct --from-speaker "Speaker 1" --name "张三"`）。

## 证据与限制

在真实测试音频（`meeting-20261008-202813-759457.wav`）上实测：原方案仅 33% 识别率且 67% 为“说话人未知”；经改造后在线识别与会后聚类达到 100% 正确归并为单一说话人。对于多人同时重叠讲话，受限于单声道单麦克风，目前仍无法实现盲源分离。

## 后果

声纹 embedding 属于敏感数据，当前只保存在本机；可通过 `meetagent speaker-correct --from-speaker` 快速全场批量更名。

