# ADR-008：音频来源

- 状态：暂定
- 日期：2026-09-29

## 决策

线下会议使用麦克风；线上会议后续支持系统音频回环，并可与麦克风混合。macOS 候选为 ScreenCaptureKit/BlackHole，Linux 候选为 PipeWire monitor source。

## 后果

音频来源作为可配置输入设备；系统音频与麦克风需保留独立来源标记。各平台采集权限和混音延迟需单独验证。
