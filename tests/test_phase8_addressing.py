"""Phase 8 P2: deciding whether an utterance is addressed to the Agent."""

import asyncio

import pytest

from meetagent.agent.addressing import AddressDetector

NAMES = ("Agent", "助手", "MeetAgent")


@pytest.mark.parametrize("text, expected", [
    ("Agent，你怎么看？", (True, "call")),
    ("Agent你怎么看", (True, "call")),           # no space between Latin and CJK
    ("助手你怎么看", (True, "call")),
    ("嘿 助手，帮我们查一下这个数据", (True, "call")),
    ("这个数字我不确定，Agent 你觉得呢", (True, "call")),
    ("Hey Agent, what do you think?", (True, "call")),
    ("MeetAgent, kannst du das erklären?", (True, "call")),
    ("我们之前问过 Agent 吗", (False, "mention")),
    ("让助手先别说话", (False, "mention")),
    ("the agents in this paper are trained offline", (False, "none")),
    ("ImageNet 有一千万张图片", (False, "none")),
])
def test_rules_without_followup_window(text, expected) -> None:
    detector = AddressDetector(NAMES)
    assert detector.rules(text, window_open=False) == expected


def test_followup_window_opens_and_expires() -> None:
    detector = AddressDetector(NAMES, followup_seconds=25)
    assert detector.rules("那子集有多少张？", window_open=False) == (False, "none")
    detector.note_agent_reply(now=100.0)
    assert detector.in_followup_window(now=120.0)
    assert not detector.in_followup_window(now=130.0)
    assert detector.rules("那子集有多少张？", window_open=True) == (True, "followup")
    # Statements in the window are not follow-ups.
    assert detector.rules("好的，我们继续下一个议题。", window_open=True) == (False, "none")


def test_prefilter_skips_classifier_for_unrelated_speech() -> None:
    calls: list[str] = []

    async def classify(names, context, utterance, window_open):
        calls.append(utterance)
        return True, "call"

    async def scenario() -> None:
        detector = AddressDetector(NAMES, classify=classify)
        decision = await detector.decide("我们下周交报告", now=0.0)
        assert decision.source == "prefilter" and not decision.addressed
        assert calls == []
        decision = await detector.decide("我们之前问过 Agent 吗", now=0.0)
        assert decision.source == "classifier" and decision.addressed
        assert calls == ["我们之前问过 Agent 吗"]

    asyncio.run(scenario())


def test_classifier_failure_falls_back_to_rules() -> None:
    async def broken(names, context, utterance, window_open):
        raise RuntimeError("quota exceeded")

    async def scenario() -> None:
        detector = AddressDetector(NAMES, classify=broken)
        decision = await detector.decide("Agent，你怎么看", now=0.0)
        assert (decision.addressed, decision.kind, decision.source) == (True, "call", "rules")
        decision = await detector.decide("我们之前问过 Agent 吗", now=0.0)
        assert (decision.addressed, decision.kind, decision.source) == (False, "mention", "rules")

    asyncio.run(scenario())


def test_clear_call_skips_classifier_and_slow_classifier_times_out() -> None:
    calls: list[str] = []

    async def slow(names, context, utterance, window_open):
        calls.append(utterance)
        await asyncio.sleep(10)
        return True, "call"

    async def scenario() -> None:
        detector = AddressDetector(NAMES, classify=slow, classify_timeout=0.05)
        decision = await detector.decide("Agent，你怎么看", now=0.0)
        assert decision.source == "rules" and decision.addressed and calls == []
        decision = await detector.decide("刚才 Agent 说的对吗", now=0.0)
        assert decision.source == "rules" and not decision.addressed
        assert calls == ["刚才 Agent 说的对吗"]

    asyncio.run(scenario())
