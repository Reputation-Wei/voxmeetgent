"""Command line interface for the demo."""

import asyncio
import webbrowser
from pathlib import Path
from typing import Literal

import typer
from rich.console import Console
from rich.markup import escape

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
    # Load the Gemini key from the system keyring when the environment has none,
    # so commands started from the bar plugin behave like the terminal.
    from meetagent.keystore import ensure_api_key_env

    ensure_api_key_env()


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


@app.command("follow")
def follow(port: int = typer.Option(8765, "--port", help="本机 MeetAgent 服务端口")) -> None:
    """实时显示正在进行的会议：带说话人的转写和 Agent 的文字回答。"""
    from meetagent.ui.follow import run_follow

    try:
        run_follow(port)
    except KeyboardInterrupt:
        pass


@app.command("calibrate-echo")
def calibrate_echo(
    write: bool = typer.Option(False, "--write", help="把推荐值写入 ~/.config/meetagent/config.toml"),
) -> None:
    """用扬声器播放一段 Agent 语音并录音，找出回声能被消除的最大音量，并推荐插话门限。"""
    from dataclasses import replace as replace_field

    from meetagent.audio.calibrate import analyze, choose_playback_gain, record_calibration, write_config

    config = AppConfig.load()
    threshold = config.live.rms_threshold
    console.print("请把系统音量调到开会时的音量，保持安静、不要碰键盘。"
                  "将用逐级降低的音量把一段 3 秒语音各播放两次，然后录 1.5 秒环境噪声…")
    levels, noise = record_calibration(delay_ms=config.audio.aec_delay_ms, threshold=threshold)
    gain, needed, results = choose_playback_gain(levels, config.audio.aec_delay_ms, threshold)
    for level_gain, level_needed in results:
        verdict = "✓" if level_needed <= 0.06 else "✗（回声太强，插话门限会过高）"
        console.print(f"  Agent 音量 ×{level_gain}: 需要插话门限 {level_needed} {verdict}")
    chosen = next((level for level in levels if level[0] == gain), levels[-1])
    result = analyze(
        chosen[1], chosen[2], noise, rms_threshold=threshold,
        current_delay_ms=config.audio.aec_delay_ms,
    )
    result = replace_field(
        result, playback_gain=gain or levels[-1][0], levels=results,
        recommended_barge_in=max(needed or result.recommended_barge_in, round(3 * result.noise_floor_rms, 3)),
    )
    if gain is None:
        result.problems.append("即使最低音量回声也无法消除；请降低系统音量后重测，或戴耳机")
    console.print(f"环境噪声 RMS：{result.noise_floor_rms:.4f}")
    console.print(f"未消除回声 RMS：{result.raw_echo_rms:.4f}（播放→麦克风延迟约 {result.measured_delay_ms:.0f} ms）")
    if result.residual_echo_p95 is not None:
        console.print(
            f"AEC 后残余回声 p95：{result.residual_echo_p95:.4f}（aec_delay_ms = {result.best_aec_delay_ms}）"
        )
    console.print(f"推荐 Agent 音量 playback_gain = {result.playback_gain}，插话门限 barge_in_threshold = {result.recommended_barge_in}")
    path = Path.home() / ".config" / "meetagent" / "config.toml"
    if result.problems:
        for problem in result.problems:
            console.print(f"[red]测量无效：{problem}[/red]")
        console.print("[red]未写入配置。[/red]")
        raise typer.Exit(1)
    if write:
        write_config(path, result)
        console.print(f"已写入 {path}")
    else:
        console.print("加 --write 可写入配置文件。")


@app.command("doctor")
def doctor(
    report_only: bool = typer.Option(
        False, "--report-only",
        help="只报告检查结果，缺少本机前提（模型、麦克风等）时也返回 0；供 CI 验证 doctor 可运行",
    ),
) -> None:
    """检查本机依赖和运行配置，不启动录音。"""
    checks = diagnostics(AppConfig.load())
    for name, ok, detail in checks:
        marker = "✓" if ok else "✗"
        # Details contain literal brackets such as "[audio].model_path".
        console.print(f"{marker} {escape(name)}: {escape(detail)}")
    raise typer.Exit(0 if report_only else exit_code(checks))


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
    time_seconds: float | None = typer.Option(None, "--time", min=0,
                                              help="JSONL timestamp 秒数；修正单条发言时使用"),
    name: str = typer.Option(..., "--name", help="正确说话人姓名或 Speaker N"),
    from_speaker: str | None = typer.Option(None, "--from-speaker",
                                            help="要全场批量重命名的原说话人标签，如 Speaker 1"),
) -> None:
    """修正说话人标注：支持按时间戳单条修正（--time）或整场批量重命名（--from-speaker）。"""
    path = transcript.expanduser()
    store = JsonlTranscriptStore(path)
    if from_speaker:
        try:
            count = store.rename_speaker(from_speaker, name)
        except Exception as exc:
            console.print(f"[red]错误：{exc}[/red]")
            raise typer.Exit(1) from exc
        console.print(f"已将全场共 {count} 条发言的说话人从“{from_speaker}”重命名为“{name}”；TXT 与 JSONL 已同步更新。")
        return
    if time_seconds is None:
        console.print("[red]错误：单条修正必须提供 --time 秒数；或提供 --from-speaker 进行批量重命名。[/red]")
        raise typer.Exit(2)
    try:
        entry = store.correct_speaker(time_seconds, name)
    except Exception as exc:
        console.print(f"[red]错误：{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(
        f"已将 [{entry['time']}] {entry['text']} 的说话人从“{entry.get('speaker_original')}”"
        f"修正为“{entry['speaker']}”；TXT 与 JSONL 已同步更新。"
    )

