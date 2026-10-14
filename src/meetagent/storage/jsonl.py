"""Durable JSONL and readable text exports for a meeting transcript."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class JsonlTranscriptStore:
    """Persist ordered transcript entries to paired TXT and JSONL files."""

    def __init__(self, text_path: Path) -> None:
        self.text_path = text_path
        self.jsonl_path = text_path.with_suffix(".jsonl")

    def append(self, entry: dict[str, Any]) -> None:
        with self.text_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"[{entry['time']}] {entry['speaker']}: {entry['text']} ({entry['language']})\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def rewrite(self, entries: list[dict[str, Any]]) -> None:
        entries.sort(key=lambda item: item["timestamp"])
        with self.text_path.open("w", encoding="utf-8") as handle:
            handle.writelines(
                f"[{entry['time']}] {entry['speaker']}: {entry['text']} ({entry['language']})\n"
                for entry in entries
            )
            handle.flush()
            os.fsync(handle.fileno())
        with self.jsonl_path.open("w", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def correct_speaker(
        self, timestamp: float, speaker: str, tolerance_seconds: float = 0.75,
    ) -> dict[str, Any]:
        """Correct the nearest utterance speaker and rewrite paired TXT/JSONL exports."""
        speaker = speaker.strip()
        if not speaker:
            raise ValueError("说话人名称不能为空。")
        entries = [
            json.loads(line)
            for line in self.jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        candidates = [
            (abs(float(entry["timestamp"]) - timestamp), entry)
            for entry in entries
            if entry.get("speaker") != "Agent"
        ]
        if not candidates:
            raise ValueError("逐字稿中没有可修正的发言。")
        distance, target = min(candidates, key=lambda item: item[0])
        if distance > tolerance_seconds:
            raise ValueError(f"最近发言与指定时间相差 {distance:.2f} 秒；拒绝改错条目。")
        old_speaker = target.get("speaker", "说话人未知")
        target["speaker"] = speaker
        target["speaker_corrected"] = True
        target["speaker_original"] = old_speaker
        self.rewrite(entries)
        return target

    def rename_speaker(self, old_speaker: str, new_speaker: str) -> int:
        """Rename all occurrences of old_speaker to new_speaker in paired TXT/JSONL exports."""
        old_speaker = old_speaker.strip()
        new_speaker = new_speaker.strip()
        if not old_speaker or not new_speaker:
            raise ValueError("说话人名称不能为空。")
        entries = [
            json.loads(line)
            for line in self.jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        count = 0
        for entry in entries:
            if entry.get("speaker") == old_speaker:
                entry["speaker"] = new_speaker
                entry["speaker_corrected"] = True
                entry["speaker_original"] = old_speaker
                count += 1
        if count > 0:
            self.rewrite(entries)
        return count

