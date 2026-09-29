"""Basic checks for Phase 1 configuration and event contracts."""

import asyncio

from meetagent.config import AppConfig
from meetagent.events import EventBus, Utterance


def test_default_config_has_documented_audio_and_output_defaults() -> None:
    config = AppConfig.load()
    assert config.audio.sample_rate == 16_000
    assert config.audio.language == "auto"
    assert config.output.save_audio is True


def test_event_bus_assigns_sequence_and_delivers_event() -> None:
    async def scenario() -> None:
        bus = EventBus()
        inbox = bus.subscribe(Utterance)
        event = await bus.publish(Utterance(
            meeting_id="test-meeting",
            start_seconds=0.0,
            end_seconds=1.0,
            text="hello",
            language="en",
        ))
        delivered = await inbox.get()
        assert delivered == event
        assert delivered.sequence == 1

    asyncio.run(scenario())


def test_event_bus_reports_backpressure() -> None:
    async def scenario() -> None:
        bus = EventBus(queue_size=1)
        inbox = bus.subscribe(Utterance)
        first = Utterance(meeting_id="test", start_seconds=0, end_seconds=1, text="one")
        second = Utterance(meeting_id="test", start_seconds=1, end_seconds=2, text="two")
        await bus.publish(first)
        try:
            await bus.publish(second)
        except asyncio.QueueFull:
            pass
        else:
            raise AssertionError("full subscriber queue must not silently drop transcript events")
        assert (await inbox.get()).text == "one"

    asyncio.run(scenario())
