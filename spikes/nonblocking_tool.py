"""Check whether a slow NON_BLOCKING Live tool stalls audio output."""

import argparse
import asyncio
import os
import time

from google import genai
from google.genai import types


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay", type=float, default=5)
    args = parser.parse_args()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("请先设置 GEMINI_API_KEY（或 GOOGLE_API_KEY）。")
    client = genai.Client(api_key=api_key)
    started = time.perf_counter()
    tool_started: float | None = None
    tool_finished = asyncio.Event()
    audio_before_tool_result = False
    function = types.FunctionDeclaration(
        name="slow_test_tool",
        description="A test tool that takes several seconds. Call it when asked to verify that the conversation remains responsive.",
        behavior="NON_BLOCKING",
        parameters={"type": "OBJECT", "properties": {}, "required": []},
    )
    config = types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        system_instruction=types.Content(parts=[types.Part(text=(
            "You are testing asynchronous tools. Call slow_test_tool when asked, "
            "say a brief spoken status while it runs, and then summarize its result."
        ))]),
        tools=[types.Tool(function_declarations=[function])],
    )
    async with client.aio.live.connect(model="gemini-3.8-live", config=config) as session:
        await session.send_client_content(
            turns=types.Content(role="user", parts=[types.Part(
                text=f"Call slow_test_tool now. It takes {args.delay:g} seconds. Tell me briefly that you're checking, then report the result."
            )]),
            turn_complete=True,
        )
        while True:
            got = 0
            complete = False
            async for response in session.receive():
                got += 1
                content = response.server_content
                if content and content.model_turn:
                    for part in content.model_turn.parts or []:
                        if getattr(getattr(part, "inline_data", None), "data", None):
                            if not tool_finished.is_set():
                                audio_before_tool_result = True
                                print(f"audio_during_tool_wait_seconds={time.perf_counter() - started:.3f}")
                if response.tool_call:
                    for call in response.tool_call.function_calls:
                        tool_started = time.perf_counter()
                        print(f"tool_started_seconds={tool_started - started:.3f}")

                        async def execute_tool(function_call: object) -> None:
                            nonlocal tool_started
                            await asyncio.sleep(args.delay)
                            tool_finished.set()
                            response_value = types.FunctionResponse(
                                id=function_call.id,
                                name=function_call.name,
                                response={"result": f"completed after {args.delay:g} seconds", "scheduling": "WHEN_IDLE"},
                            )
                            await session.send_tool_response(function_responses=[response_value])
                            print(f"tool_finished_seconds={time.perf_counter() - started:.3f}")

                        asyncio.create_task(execute_tool(call), name="slow-nonblocking-tool")
                if content and content.turn_complete:
                    complete = True
                    break
            if got == 0 or complete:
                break
    print(f"audio_received_before_tool_result={audio_before_tool_result}")
    print(f"tool_delay_seconds={args.delay:g}")


if __name__ == "__main__":
    asyncio.run(main())
