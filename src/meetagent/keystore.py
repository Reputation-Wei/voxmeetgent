"""Gemini API key storage in the system keyring (libsecret on Linux).

The key never goes into config.toml, shell.json or any other plain file.
Lookup order: GEMINI_API_KEY / GOOGLE_API_KEY environment variables, then
the keyring entry `service=gemini user=meetagent` (the same entry created by
`secret-tool store --label=... service gemini user meetagent`).

Why the keyring and not only the environment: Omarchy's ~/.bashrc returns
early for non-interactive shells, so a service started from the bar plugin
never sees an `export GEMINI_API_KEY=...` placed later in that file.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess

KEYRING_ATTRIBUTES = ("service", "gemini", "user", "meetagent")
ENV_NAMES = ("GEMINI_API_KEY", "GOOGLE_API_KEY")


def env_api_key() -> str | None:
    for name in ENV_NAMES:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def keyring_api_key() -> str | None:
    """Read the key from libsecret; None if unavailable or not stored."""
    if platform.system() != "Linux" or shutil.which("secret-tool") is None:
        return None
    try:
        result = subprocess.run(
            ["secret-tool", "lookup", *KEYRING_ATTRIBUTES],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    value = result.stdout.strip()
    return value or None


def ensure_api_key_env() -> bool:
    """Put the keyring key into GEMINI_API_KEY if no key is set; return availability.

    Every Gemini call in MeetAgent reads the environment, so loading it once at
    process start makes CLI, GUI service and plugin-started service behave alike.
    """
    if env_api_key():
        return True
    key = keyring_api_key()
    if key:
        os.environ["GEMINI_API_KEY"] = key
        return True
    return False


def store_api_key(key: str) -> None:
    """Save the key in the keyring and use it in this process from now on."""
    key = key.strip()
    if not key or any(ch.isspace() for ch in key):
        raise ValueError("API key 不能为空，也不能包含空格或换行。")
    if platform.system() != "Linux" or shutil.which("secret-tool") is None:
        raise RuntimeError("当前系统没有 secret-tool；请改用环境变量 GEMINI_API_KEY。")
    # The key goes through stdin, never the command line (visible in `ps`).
    result = subprocess.run(
        ["secret-tool", "store", "--label=Gemini API key (MeetAgent)", *KEYRING_ATTRIBUTES],
        input=key, capture_output=True, text=True, timeout=15, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"写入系统钥匙环失败：{result.stderr.strip() or result.returncode}")
    os.environ["GEMINI_API_KEY"] = key
    os.environ.pop("GOOGLE_API_KEY", None)  # the new key wins over an older one


async def validate_api_key(key: str) -> None:
    """Raise ValueError if Gemini rejects the key (a free metadata request)."""
    from google import genai

    try:
        await genai.Client(api_key=key.strip()).aio.models.get(model="gemini-3.8-flash")
    except Exception as exc:
        message = str(exc)
        if "API key not valid" in message or "PERMISSION_DENIED" in message or "401" in message or "403" in message:
            raise ValueError("Gemini 拒绝了这个 API key，请确认复制的是 AI Studio 的完整 key。") from exc
        raise RuntimeError(f"暂时无法验证 API key：{message}") from exc
