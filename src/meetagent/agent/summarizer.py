"""Source-grounded multilingual meeting summary generation and persistence."""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import date
from pathlib import Path
from typing import Any, Protocol


class Summarizer(Protocol):
    """Create a source-grounded Markdown meeting summary."""

    async def summarize(self, events: list[dict[str, Any]], language: str = "zh") -> str:
        """Return Markdown without inventing decisions or action items."""
        ...


def _transcript(events: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"[{event.get('time', '--:--')}] {event.get('speaker', '说话人未知')}: {event.get('text', '')}"
        for event in sorted(events, key=lambda item: float(item.get("timestamp", 0)))
        if event.get("text")
    )


def _chunks(text: str, limit: int = 24_000) -> list[str]:
    """Split long transcripts at utterance boundaries for predictable API payloads."""
    lines = text.splitlines()
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for line in lines:
        if current and size + len(line) + 1 > limit:
            chunks.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


# No client-side tools here; disabling AFC also silences the SDK warning.
_NO_AFC = {"automatic_function_calling": {"disable": True}}


def _language_name(language: str) -> str:
    return {"zh": "简体中文", "en": "English", "de": "Deutsch"}[language]


async def summarize_with_gemini(
    events: list[dict[str, Any]], answers: list[dict[str, Any]], language: str = "zh",
    findings: list[dict[str, Any]] | None = None,
) -> str:
    """Summarize each transcript chunk, then consolidate into the requested language."""
    if language not in {"zh", "en", "de"}:
        raise ValueError("纪要语言必须为 zh、en 或 de。")
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        raise RuntimeError("未设置 GEMINI_API_KEY 或 GOOGLE_API_KEY")
    from google import genai

    client = genai.Client()
    transcript = _transcript(events)
    chunks = _chunks(transcript)
    if not chunks:
        chunks = ["（没有识别到会议发言）"]
    partials: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        prompt = (
            f"用{_language_name(language)}从逐字稿第 {index}/{len(chunks)} 部分提取：讨论议题、明确决议、明确待办（负责人/截止日期；未提及写未指定）、"
            "事实核查结论、研究方向、未解决问题。严格依据原文；不确定性明确标出；不得推断未说出的决议或负责人。"
            "每个要点保留时间戳和说话人。输出简洁 Markdown。\n\n逐字稿：\n"
            f"{chunk}"
        )
        response = await client.aio.models.generate_content(
            model="gemini-3.8-flash", contents=prompt, config=_NO_AFC,
        )
        if not response.text:
            raise RuntimeError(f"摘要分块 {index} 返回空内容。")
        partials.append(response.text)

    answers_json = json.dumps(answers, ensure_ascii=False)
    findings_json = json.dumps(findings or [], ensure_ascii=False)
    if len(partials) == 1:
        consolidation_source = partials[0]
    else:
        consolidation_source = "\n\n".join(
            f"## 逐字稿分块 {index}\n{part}" for index, part in enumerate(partials, start=1)
        )
    final_prompt = (
        f"将以下逐字稿分块提要合并为一份完整的{_language_name(language)}会议纪要。\n"
        "必须包含：摘要、按议题整理的讨论要点、决议、待办（负责人/截止日期）、事实核查记录、研究方向、未解决问题、Agent 回答及来源。"
        "消除重复；时间戳和人名以输入为准；没有明确内容时写‘未记录’，不要编造。来源 URL 原样保留。\n\n"
        f"分块提要：\n{consolidation_source}\n\n事实核查/研究 Finding 记录（JSON）：\n{findings_json}"
        f"\n\nAgent 回答记录（JSON）：\n{answers_json}"
    )
    result = await client.aio.models.generate_content(
        model="gemini-3.8-flash", contents=final_prompt, config=_NO_AFC,
    )
    if not result.text:
        raise RuntimeError("最终纪要返回空内容。")
    return result.text.strip()


def persist_summary(
    body: str, transcript_path: Path, *, meeting_date: date | None = None,
    vault: Path | None = None, vault_folder: str = "Meetings",
) -> Path:
    """Atomically write Markdown beside transcripts and optionally into an Obsidian vault."""
    meeting_date = meeting_date or date.today()
    safe_folder = Path(vault_folder)
    if safe_folder.is_absolute() or ".." in safe_folder.parts:
        raise ValueError("Obsidian 目录必须是 vault 内的相对路径。")
    frontmatter = (
        f"---\ndate: {meeting_date.isoformat()}\nparticipants: []\ntags: [meeting]\n---\n\n"
    )
    markdown = body.strip()
    if not markdown.startswith("---\n"):
        markdown = frontmatter + markdown + "\n"
    path = transcript_path.with_suffix(".md")
    _atomic_write(path, markdown)
    if vault is not None:
        try:
            vault = vault.expanduser().resolve()
            if not (vault / ".obsidian").is_dir():
                raise ValueError(f"配置的 Obsidian 路径不是 vault（缺少 .obsidian）：{vault}")
            folder = (vault / safe_folder).resolve()
            if vault not in folder.parents and folder != vault:
                raise ValueError("Obsidian 纪要目录必须位于 vault 内。")
            folder.mkdir(parents=True, exist_ok=True)
            vault_path = folder / path.name
            _atomic_write(vault_path, markdown)
            return vault_path
        except Exception as exc:
            raise RuntimeError(f"本地纪要已保存到 {path}，但 Obsidian 写入失败：{exc}") from exc
    return path


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def local_fallback_summary(
    events: list[dict[str, Any]], answers: list[dict[str, Any]],
    language: str = "zh", findings: list[dict[str, Any]] | None = None,
) -> str:
    """Create a truthful, transcript-based summary when the model is unavailable."""
    headings = {
        "zh": ("会议纪要", "摘要", "讨论要点", "请根据以下逐字稿人工整理。", "决议", "未记录（未完成 Gemini 摘要）。", "待办", "事实核查记录", "研究方向", "未解决问题", "逐字稿", "Agent 回答及来源", "本地逐字稿已保存。Gemini 摘要不可用；以下仅列出识别记录，未生成模型归纳。"),
        "en": ("Meeting notes", "Summary", "Discussion", "Review the transcript below manually.", "Decisions", "Not recorded (Gemini summary unavailable).", "Action items", "Fact checks", "Research directions", "Open questions", "Transcript", "Agent responses and sources", "The local transcript was saved. Gemini summary was unavailable; no model-generated synthesis is included."),
        "de": ("Besprechungsnotizen", "Zusammenfassung", "Diskussion", "Bitte das folgende Transkript manuell auswerten.", "Beschlüsse", "Nicht protokolliert (Gemini-Zusammenfassung nicht verfügbar).", "Aufgaben", "Faktenprüfungen", "Forschungsrichtungen", "Offene Fragen", "Transkript", "Antworten und Quellen des Agenten", "Das lokale Transkript wurde gespeichert. Gemini war nicht verfügbar; es wurde keine modellgenerierte Zusammenfassung erstellt."),
    }
    title, summary, discussion, review_text, decisions, missing, actions, fact_checks, research, questions, transcript_title, agent_title, unavailable = headings[language]
    lines = [f"# {title}", "", f"## {summary}", "", unavailable, "", f"## {discussion}", "", review_text,
             "", f"## {decisions}", "", f"- {missing}", "", f"## {actions}", "", f"- {missing}",
             "", f"## {fact_checks}", ""]
    if findings:
        lines.extend(f"- [{item.get('kind', 'finding')}] {item.get('claim', '')} — {item.get('verdict', 'unverified')}" for item in findings)
    else:
        lines.append("- 未记录。" if language == "zh" else ("- Not recorded." if language == "en" else "- Nicht protokolliert."))
    lines.extend(["", f"## {research}", "", "- 未记录。" if language == "zh" else ("- Not recorded." if language == "en" else "- Nicht protokolliert."),
                  "", f"## {questions}", "", "- 未记录。" if language == "zh" else ("- Not recorded." if language == "en" else "- Nicht protokolliert."),
                  "", f"## {transcript_title}", ""])
    lines.extend(
        f"- [{event.get('time', '--:--')}] {event.get('speaker', '说话人未知')}: {event.get('text', '')}"
        for event in sorted(events, key=lambda item: float(item.get("timestamp", 0)))
    )
    lines.extend(["", f"## {agent_title}", ""])
    if answers:
        for answer in answers:
            lines.extend([f"### {answer.get('trigger', '回答')}", "", answer.get("answer", ""), ""])
            lines.extend(f"- 来源：{url}" for url in answer.get("sources", []))
    else:
        lines.append("- 未记录 Agent 回答。" if language == "zh" else ("- No Agent responses recorded." if language == "en" else "- Keine Agent-Antworten protokolliert."))
    return "\n".join(lines).rstrip() + "\n"
