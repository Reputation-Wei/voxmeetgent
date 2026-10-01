"""Phase 8 P1: shared meeting state, keyword triggers, and speech authorization."""

import asyncio
from pathlib import Path
from typing import Any

import numpy as np

from meetagent import demo
from meetagent.agent.analyst import AnalysisResult
from meetagent.agent.intervention import InterventionPolicy
from meetagent.agent.meeting_state import MeetingState
from meetagent.config import LiveConfig

NOW = 10_000.0


def _entry(offset: float, speaker: str, text: str, source: str = "local_asr") -> dict[str, Any]:
    seconds = int(offset)
    return {
        "timestamp": NOW - 600 + offset,
        "time": f"{seconds // 60:02d}:{seconds % 60:02d}",
        "speaker": speaker, "text": text, "source": source,
    }


def _meeting() -> list[dict[str, Any]]:
    return [
        _entry(10, "张三", "我们先讨论数据集。"),
        _entry(100, "李四", "ImageNet 有一千万张图片。"),
        _entry(200, "Agent", "上次的核查结论是……", source="gemini_live"),
        _entry(550, "张三", "这个结论是真的吗？"),
    ]


def test_recent_transcript_keeps_speakers_and_skips_agent() -> None:
    state = MeetingState(_meeting())
    recent = state.recent_transcript(seconds=120, now=NOW)
    assert recent == "[09:10] 张三: 这个结论是真的吗？"
    everything = state.recent_transcript(seconds=10_000, now=NOW)
    assert "李四: ImageNet" in everything
    assert "Agent" not in everything


def test_digest_folds_only_older_lines_once() -> None:
    calls: list[tuple[str, str]] = []

    async def fake_digest(previous: str, new_lines: str) -> str:
        calls.append((previous, new_lines))
        return previous + f"摘要{len(calls)}"

    async def scenario() -> None:
        state = MeetingState(_meeting(), digest_function=fake_digest)
        context = await state.analysis_context(now=NOW)
        # Older attendee lines are digested; the last two minutes stay verbatim.
        assert len(calls) == 1
        assert "张三: 我们先讨论数据集" in calls[0][1]
        assert "李四: ImageNet" in calls[0][1]
        assert "这个结论是真的吗" not in calls[0][1]
        assert "【较早讨论摘要】\n摘要1" in context
        assert "[09:10] 张三: 这个结论是真的吗？" in context
        # Nothing new became old enough, so no second Gemini call.
        await state.analysis_context(now=NOW)
        assert len(calls) == 1

    asyncio.run(scenario())


def test_live_updates_advance_cursor() -> None:
    state = MeetingState(_meeting())
    text, cursor = state.live_updates_since(0.0)
    assert text.splitlines()[0] == "[00:10] 张三: 我们先讨论数据集。"
    assert "Agent" not in text
    assert state.live_updates_since(cursor) == ("", cursor)


def test_policy_paths() -> None:
    normal = InterventionPolicy(LiveConfig())
    assert normal.may_speak("call")
    assert normal.may_speak("keyword")
    assert not normal.may_speak("hand_raise")
    assert normal.may_speak("hand_raise", approved=True)
    assert normal.authorize_live_turn(True) and not normal.authorize_live_turn(False)
    silent = InterventionPolicy(LiveConfig(intervention="silent"))
    assert not any([
        silent.may_speak("call"), silent.may_speak("keyword"),
        silent.may_speak("hand_raise", approved=True),
    ])


def test_trilingual_keywords() -> None:
    keywords = demo._load_keywords()
    assert demo._trigger("能不能帮我们总结一下刚才的讨论", keywords) == "summarize"
    assert demo._trigger("Kannst du das zusammenfassen?", keywords) == "summarize"
    assert demo._trigger("Is that true, the number you said?", keywords) == "fact_check"
    assert demo._trigger("有没有人做过类似的工作", keywords) == "research"
    assert demo._trigger("Hat das schon jemand untersucht?", keywords) == "research"
    # Mentioning the Agent alone is an addressing cue, not a request.
    assert demo._trigger("agent 在听吗", keywords) is None


class _FakeAnalyst:
    def __init__(self) -> None:
        self.contexts: list[str] = []
        self.release = asyncio.Event()

    async def respond(self, transcript: str, trigger: str, request_text: str = "") -> AnalysisResult:
        self.contexts.append(transcript)
        await self.release.wait()
        return AnalysisResult(
            answer="## 事实核查\n- 李四的数字偏高", sources=["https://example.org/imagenet"],
            papers=[], details={"spoken": "李四说的一千万张不准确，大约一百四十万张。"}, findings=[],
        )


def test_keyword_runs_in_background_and_speaks_without_name_call(tmp_path: Path, monkeypatch) -> None:
    spoken: list[str] = []

    async def fake_speak(answer: str, playback: Any = None) -> None:
        spoken.append(answer)

    monkeypatch.setattr(demo, "_speak", fake_speak)
    monkeypatch.setattr(demo, "_transcribe", lambda audio, language: ("ImageNet 这是真的吗", "zh"))

    async def scenario() -> None:
        entries = _meeting()
        for entry in entries:
            entry["timestamp"] = entry["timestamp"] - NOW + __import__("time").time()
        state = MeetingState(entries)
        analyst = _FakeAnalyst()
        tasks: set[asyncio.Task[None]] = set()
        transcript_path = tmp_path / "meeting.txt"
        await demo._process_audio(
            np.zeros(16_000, dtype=np.float32), 590.0, entries, [], demo._load_keywords(),
            __import__("time").time() - 600, transcript_path, "zh",
            analyst=analyst, policy=InterventionPolicy(LiveConfig()),
            meeting_state=state, trigger_tasks=tasks,
        )
        # ASR returned while the analysis is still running in the background.
        assert len(tasks) == 1 and not spoken
        analyst.release.set()
        await asyncio.gather(*tasks)
        assert spoken == ["李四说的一千万张不准确，大约一百四十万张。"]
        assert "李四: ImageNet 有一千万张图片" in analyst.contexts[0]
        assert state.conclusions[-1].text.startswith("李四说的")

    asyncio.run(scenario())


def test_silent_mode_never_speaks(tmp_path: Path, monkeypatch) -> None:
    spoken: list[str] = []

    async def fake_speak(answer: str, playback: Any = None) -> None:
        spoken.append(answer)

    monkeypatch.setattr(demo, "_speak", fake_speak)

    async def scenario() -> None:
        analyst = _FakeAnalyst()
        analyst.release.set()
        await demo._handle_trigger(
            "fact_check", _meeting(), [], analyst=analyst,
            policy=InterventionPolicy(LiveConfig(intervention="silent")),
            request_text="这是真的吗",
        )
        assert spoken == []

    asyncio.run(scenario())


def test_digest_failure_falls_back_to_older_verbatim_lines() -> None:
    async def broken_digest(previous: str, new_lines: str) -> str:
        raise RuntimeError("network down")

    async def scenario() -> None:
        state = MeetingState(_meeting(), digest_function=broken_digest)
        context = await state.analysis_context(now=NOW)
        assert "【较早讨论（原文）】" in context
        assert "李四: ImageNet 有一千万张图片" in context

    asyncio.run(scenario())


def test_chinese_call_accepts_zhushou() -> None:
    assert demo._contains_authorized_call("助手，你怎么看？")
    assert demo._contains_authorized_call("Agent, what do you think?")
    assert not demo._contains_authorized_call("你怎么看")


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _audio() -> bytes:
    return np.full(240, 1000, dtype="<i2").tobytes()


def test_call_survives_interrupt_caused_by_the_call_itself() -> None:
    clock = _Clock()
    playback = demo._PlaybackBuffer()
    gate = demo._LiveCallGate(playback, clock=clock)
    gate.call_sent()
    # Live's own reply to the heard audio starts, then our text interrupts it.
    gate.on_audio(_audio())
    clock.now = 1.0
    assert gate.on_interrupted() is True
    # The old turn reports completion before the real answer starts.
    gate.on_turn_complete()
    assert playback.response_authorized()
    clock.now = 2.0
    gate.on_audio(_audio())
    assert playback.is_speaking()
    gate.on_turn_complete()
    assert not playback.response_authorized() and not gate.waiting


def test_late_interrupt_is_a_real_barge_in() -> None:
    clock = _Clock()
    playback = demo._PlaybackBuffer()
    gate = demo._LiveCallGate(playback, clock=clock)
    gate.call_sent()
    clock.now = 5.0
    assert gate.on_interrupted() is False
    assert not playback.response_authorized() and not gate.waiting


def test_unanswered_grant_expires() -> None:
    clock = _Clock()
    playback = demo._PlaybackBuffer()
    gate = demo._LiveCallGate(playback, clock=clock)
    gate.call_sent()
    clock.now = 20.0
    gate.on_audio(_audio())  # unrelated speech much later must not play
    assert not playback.is_speaking() and not gate.waiting


def test_text_reply_mode_never_speaks() -> None:
    policy = InterventionPolicy(LiveConfig(reply="text"))
    assert not policy.may_speak("call")
    assert not policy.may_speak("keyword")
    assert not policy.authorize_live_turn(True)


def test_name_call_without_live_is_answered_by_analyst(tmp_path: Path, monkeypatch) -> None:
    spoken: list[str] = []

    async def fake_speak(answer: str, playback: Any = None) -> None:
        spoken.append(answer)

    monkeypatch.setattr(demo, "_speak", fake_speak)
    monkeypatch.setattr(demo, "_transcribe", lambda audio, language: ("助手你怎么看", "zh"))
    seen: list[str] = []

    class Analyst(_FakeAnalyst):
        async def respond(self, transcript: str, trigger: str, request_text: str = "") -> AnalysisResult:
            seen.append(trigger)
            return AnalysisResult("ImageNet 约 1400 万张。", [], [], {"answer": "ImageNet 约 1400 万张。"}, [])

    async def scenario() -> None:
        answers: list[dict[str, Any]] = []
        await demo._process_audio(
            np.zeros(16_000, dtype=np.float32), 5.0, [], answers, demo._load_keywords(),
            __import__("time").time(), tmp_path / "m.txt", "zh",
            analyst=Analyst(), policy=InterventionPolicy(LiveConfig(reply="text")),
            live_calls=None,
        )
        assert seen == ["ask"]
        assert answers[0]["answer"] == "ImageNet 约 1400 万张。"
        assert spoken == []

    asyncio.run(scenario())


def test_service_reports_latest_agent_answer(tmp_path: Path) -> None:
    import json

    from meetagent.ui.service import MeetingService

    path = tmp_path / "m.jsonl"
    rows = [
        {"time": "00:01", "speaker": "张三", "text": "ImageNet 有一千万张"},
        {"time": "00:05", "speaker": "Agent", "text": "旧回答", "source": "analyst"},
        {"time": "00:09", "speaker": "Agent", "text": "约 1400 万张", "source": "analyst"},
        {"time": "00:12", "speaker": "李四", "text": "好的"},
    ]
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    assert MeetingService._last_agent_entry(path)["text"] == "约 1400 万张"
    assert MeetingService._last_agent_entry(tmp_path / "missing.jsonl") is None
