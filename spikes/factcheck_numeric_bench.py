"""Numeric fact-check benchmark against the real Gemini API (costs a few cents).

Checks that clearly wrong numbers are not judged "supported" just because they
are the same order of magnitude, and that correct numbers stay "supported".
Includes the noisy in-meeting context from the 2026-10-01 Omarchy test, run
5 times to expose run-to-run variance.

    GEMINI_API_KEY=... uv run python spikes/factcheck_numeric_bench.py
"""
import asyncio
import collections
from meetagent import demo
CASES = [
    ("ImageNet 有1000万张图片", {"contradicted", "mixed"}),
    ("ImageNet 大约有一千四百万张图片", {"supported"}),
    ("ImageNet-1K 的训练集大约有128万张图片", {"supported"}),
    ("珠穆朗玛峰海拔是9000米", {"contradicted"}),
    ("GPT-3 有1750亿个参数", {"supported"}),
    ("地球到月球的平均距离大约是38万公里", {"supported"}),
    ("人类基因组大约有10万个蛋白编码基因", {"contradicted"}),
]
async def one(text, ok):
    transcript = f"【最近逐字稿（带说话人）】\n[00:05] 李四: {text}\n[00:09] 张三: 这是真的吗"
    _, _, _, d = await demo._answer(transcript, "fact_check", "这是真的吗")
    f = (d.get("findings") or [{}])[0]
    v = f.get("verdict", "?")
    extra = f" stated={f.get('stated_value','')} ref={f.get('reference_value','')}" if "stated_value" in f else ""
    print(f"{'PASS' if v in ok else 'FAIL'} {v:12s} {text}{extra}\n     spoken: {d.get('spoken','')}")
    return v in ok
CTX = ("【最近逐字稿（带说话人）】\n[00:04] 说话人未知: 因为这Net有1000万张图片\n"
       "[00:08] Speaker: 也门胡塞武装有 1000 万人突变。\n[00:11] Speaker 1: ImageNet有1000万张图片\n"
       "[00:17] 说话人未知: 这是真的吗")
async def meeting_one():
    _, _, _, d = await demo._answer(CTX, "fact_check", "这是真的吗")
    fs = d.get("findings") or []
    f = next((x for x in fs if "ImageNet" in x.get("claim","") or "图片" in x.get("claim","")), fs[0] if fs else {})
    return f.get("verdict", "?")


async def main():
    results = await asyncio.gather(*(one(t, ok) for t, ok in CASES))
    print(f"{sum(results)}/{len(results)} as expected")
    verdicts = await asyncio.gather(*(meeting_one() for _ in range(5)))
    print("noisy meeting context, 1000万 claim:", collections.Counter(verdicts))


asyncio.run(main())
