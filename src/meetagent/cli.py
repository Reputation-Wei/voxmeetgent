"""Command line interface for the demo."""

import asyncio
import webbrowser
from pathlib import Path
from typing import Literal

import typer
from rich.console import Console

from meetagent.demo import run_demo, run_local_transcription
from meetagent.config import AppConfig
from meetagent.doctor import diagnostics, exit_code
from meetagent.diarization.sherpa import enroll_speaker
from meetagent.storage.jsonl import JsonlTranscriptStore
from meetagent.ui.server import create_app

app = typer.Typer(
    no_args_is_help=True,
    help="本地会议助手最小 Demo",
    invoke_without_command=False,
    pretty_exceptions_enable=False,
)
gui_app = typer.Typer(no_args_is_help=True, help="启动仅本机可访问的 MeetAgent 图形界面")
app.add_typer(gui_app, name="gui")
console = Console()


@app.callback()
def main() -> None:
    """Run a MeetAgent command."""


@gui_app.callback(invoke_without_command=True)
def gui(
    ctx: typer.Context,
    port: int = typer.Option(8765, "--port", min=1024, max=65535, help="本机 Web GUI 端口"),
    no_open: bool = typer.Option(False, "--no-open", help="不自动打开浏览器"),
) -> None:
    """启动本机 GUI 服务并在浏览器打开会议控制台。"""
    if ctx.invoked_subcommand:
        return
    import uvicorn

    url = f"http://127.0.0.1:{port}/"
    console.print(f"MeetAgent GUI：{url}")
    console.print("服务仅监听本机；关闭此终端会结束 GUI 服务。")
    if not no_open:
        webbrowser.open(url)
    uvicorn.run(create_app(), host="127.0.0.1", port=port, log_level="warning")


@app.command("doctor")
def doctor() -> None:
    """检查本机依赖和运行配置，不启动录音。"""
    checks = diagnostics(AppConfig.load())
    for name, ok, detail in checks:
        marker = "✓" if ok else "✗"
        console.print(f"{marker} {name}: {detail}")
    raise typer.Exit(exit_code(checks))


@app.command("demo")
def demo(
    out: Path | None = typer.Option(None, "--out", help="纪要和转写的输出目录"),
    file: Path | None = typer.Option(None, "--file", help="离线 WAV 文件；提供时不打开麦克风"),
    language: Literal["auto", "zh", "en", "de"] = typer.Option(
        "auto", "--language", help="Whisper 识别语言：auto 自动检测，或指定 zh / en / de",
    ),
    summary_language: Literal["zh", "en", "de"] = typer.Option(
        "zh", "--summary-language", help="会后纪要语言：zh / en / de",
    ),
    obsidian_vault: Path | None = typer.Option(
        None, "--obsidian-vault", help="可选 Obsidian vault 路径；纪要另写入 vault 指定文件夹",
    ),
    obsidian_folder: str = typer.Option(
        "Meetings", "--obsidian-folder", help="vault 内的纪要相对目录，默认 Meetings",
    ),
    consent_to_stream: bool = typer.Option(
        False, "--consent-to-stream",
        help="确认与会者同意将整场麦克风音频实时发送到 Gemini Live",
    ),
    analyze_claims: bool = typer.Option(
        False, "--analyze-claims",
        help="额外同意每分钟将最近逐字稿发送到 Gemini Flash，静默抽取可核查陈述",
    ),
    intervention: Literal["silent", "on_request", "proactive"] | None = typer.Option(
        None, "--intervention", help="覆盖介入策略；所有模式都必须先明确呼叫 Agent",
    ),
    aec: bool | None = typer.Option(
        None, "--aec/--no-aec", help="启用或关闭声学回声消除；默认读取配置",
    ),
    reply: Literal["voice", "text"] | None = typer.Option(
        None, "--reply",
        help="voice 语音回答；text 只打印文字回答且不连接 Gemini Live（不上传会议音频）。默认读取配置",
    ),
) -> None:
    """启动持续倾听、被点名语音问答、本地转写和会后总结。"""
    if out is None:
        typer.echo("错误：必须指定 --out <文件夹>。Demo 结束时会在此保存纪要和转写。", err=True)
        raise typer.Exit(2)
    if file is not None and not file.is_file():
        typer.echo(f"错误：找不到音频文件：{file}", err=True)
        raise typer.Exit(2)
    if file is not None and analyze_claims:
        typer.echo("错误：--analyze-claims 仅适用于实时麦克风会议。", err=True)
        raise typer.Exit(2)
    has_gemini_key = bool(
        __import__("os").environ.get("GEMINI_API_KEY")
        or __import__("os").environ.get("GOOGLE_API_KEY")
    )
    text_reply = (reply or AppConfig.load().live.reply) == "text"
    if not file and has_gemini_key and not consent_to_stream and not text_reply:
        typer.echo(
            "错误：实时模式会将整场麦克风音频发送到 Gemini Live。取得与会者同意后，请添加 --consent-to-stream。",
            err=True,
        )
        raise typer.Exit(2)
    if not file and not has_gemini_key:
        typer.echo("未发现 Gemini API key：将只进行本地录音和转写，不上传音频。", err=True)
    try:
        asyncio.run(run_demo(
            out.expanduser(), file.expanduser() if file else None, language=language,
            analyze_claims=analyze_claims, intervention=intervention, aec_enabled=aec,
            summary_language=summary_language,
            obsidian_vault=obsidian_vault.expanduser() if obsidian_vault else None,
            obsidian_folder=obsidian_folder, reply=reply,
        ))
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        console.print(f"[red]错误：{exc}[/red]")
        raise typer.Exit(1) from exc


@app.command("transcribe")
def transcribe(
    file: Path = typer.Option(..., "--file", exists=True, dir_okay=False, help="本地 16-bit PCM WAV 文件"),
    out: Path = typer.Option(..., "--out", help="TXT、JSONL、Markdown 和 WAV 的输出目录"),
    language: Literal["auto", "zh", "en", "de"] = typer.Option(
        "auto", "--language", help="Whisper 识别语言：auto 自动检测，或指定 zh / en / de",
    ),
) -> None:
    """仅用本地 VAD + Whisper 转写 WAV，不连接 Gemini。"""
    try:
        asyncio.run(run_local_transcription(file.expanduser(), out.expanduser(), language))
    except Exception as exc:
        console.print(f"[red]错误：{exc}[/red]")
        raise typer.Exit(1) from exc


@app.command("enroll")
def enroll(
    name: str = typer.Option(..., "--name", help="与会者姓名或自选显示名称"),
    file: Path = typer.Option(..., "--file", exists=True, dir_okay=False, help="单人 16-bit PCM WAV 声纹样本"),
) -> None:
    """本地注册说话人声纹，仅保存 embedding，不保留原始样本。"""
    try:
        registry_path = enroll_speaker(name, file.expanduser())
    except Exception as exc:
        console.print(f"[red]错误：{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(f"已注册说话人“{name}”；声纹保存在本地：{registry_path}")


@app.command("speaker-correct")
def speaker_correct(
    transcript: Path = typer.Option(..., "--transcript", exists=True, dir_okay=False,
                                    help="会议 .txt 文件；对应 .jsonl 也必须存在"),
    time_seconds: float = typer.Option(..., "--time", min=0,
                                       help="JSONL timestamp 秒数；可从 JSONL 的 timestamp 字段复制"),
    name: str = typer.Option(..., "--name", help="正确说话人姓名或 Speaker N"),
) -> None:
    """修正最近的一条说话人标注，并同步更新 TXT/JSONL。"""
    path = transcript.expanduser()
    try:
        entry = JsonlTranscriptStore(path).correct_speaker(time_seconds, name)
    except Exception as exc:
        console.print(f"[red]错误：{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(
        f"已将 [{entry['time']}] {entry['text']} 的说话人从“{entry.get('speaker_original')}”"
        f"修正为“{entry['speaker']}”；TXT 与 JSONL 已同步更新。"
    )
