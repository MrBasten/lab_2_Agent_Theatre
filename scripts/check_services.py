"""Check local configuration, or explicitly test one connected service.

The default mode performs no network requests and never imports torch. Keys are
read only from the process environment or the project's local .env file.
Voice endpoint: https://elevenlabs.io/docs/api-reference/voices/search
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from theatre.audio import AudioError, render_audio  # noqa: E402
from theatre.providers import CloudProvider, ProviderError  # noqa: E402


class ServiceCheckError(RuntimeError):
    """Safe diagnostic text, without server response bodies or credentials."""


def is_configured(value: str | None) -> bool:
    return bool(value and value.strip() and value.strip() not in {"...", "YOUR_API_KEY", "<API_key>"})


def _safe_text(value: object) -> str:
    """Keep even unusual service-provided names from printing configured keys."""
    text = str(value)
    for variable in ("LLM_API_KEY", "YANDEX_API_KEY", "api_key", "ELEVENLABS_API_KEY"):
        secret = os.getenv(variable)
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return "".join(character if character.isprintable() else " " for character in text)


def offline_report() -> None:
    print("Offline configuration check: no network requests; secret values are hidden.")
    print(f".env file: {'PRESENT' if (ROOT / '.env').is_file() else 'ABSENT'}")
    groups = (
        (
            "Cloud API key (LLM_API_KEY / YANDEX_API_KEY / api_key)",
            ("LLM_API_KEY", "YANDEX_API_KEY", "api_key"),
        ),
        ("Yandex folder (YANDEX_FOLDER_ID / folder_id)", ("YANDEX_FOLDER_ID", "folder_id")),
        ("LLM_MODEL (optional Yandex default)", ("LLM_MODEL",)),
        ("LLM_BASE_URL (optional Yandex default)", ("LLM_BASE_URL",)),
        ("LLM_API_KIND (optional provider default)", ("LLM_API_KIND",)),
        ("ELEVENLABS_API_KEY", ("ELEVENLABS_API_KEY",)),
        *(
            (f"ELEVENLABS_VOICE_{role}", (f"ELEVENLABS_VOICE_{role}",))
            for role in ("VERA", "MARK", "ECHO", "NARRATOR")
        ),
    )
    for label, variables in groups:
        print(f"{label}: {'SET' if any(is_configured(os.getenv(name)) for name in variables) else 'UNSET'}")
    for module in ("torch", "transformers", "accelerate", "bitsandbytes", "dotenv"):
        print(
            f"Python package {module}: {'INSTALLED' if importlib.util.find_spec(module) is not None else 'MISSING'}"
        )
    print(f"FFmpeg: {'AVAILABLE' if shutil.which('ffmpeg') else 'MISSING (individual clips still work)'}")
    print("CUDA: NOT PROBED. Run scripts/check_local_model.py to test actual GPU inference.")
    print(
        "Configuration presence does not prove service access. Use --cloud, --voices, or --audio-smoke explicitly."
    )


def check_cloud() -> None:
    provider = CloudProvider.from_env()
    # A smoke check is one small request, with no automatic retry billing.
    provider.max_attempts = 1
    provider.timeout = min(provider.timeout, 60)
    text = provider.complete(
        [
            {"role": "system", "content": "Ответь одной короткой фразой по-русски."},
            {"role": "user", "content": "Поздоровайся с посетителем театра."},
        ],
        temperature=0.2,
        max_tokens=100,
    )
    if not text.strip():
        raise ServiceCheckError("Cloud returned an empty response.")
    # Do not print an untrusted response or the request headers.
    print("Cloud check: OK; one real text response received (max_tokens=100).")


def list_voices() -> list[dict]:
    key = os.getenv("ELEVENLABS_API_KEY", "")
    if not is_configured(key):
        raise ServiceCheckError("Set ELEVENLABS_API_KEY in the local .env first.")
    token: str | None = None
    seen_tokens: set[str] = set()
    voices: dict[str, dict] = {}
    # Official /v2/voices uses token pagination, with <=100 items per page.
    for _ in range(20):
        query = {"page_size": "100", "include_total_count": "false"}
        if token:
            query["next_page_token"] = token
        request = urllib.request.Request(
            "https://api.elevenlabs.io/v2/voices?" + urllib.parse.urlencode(query),
            headers={"xi-api-key": key, "Accept": "application/json"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read(8_000_001)
            if len(raw) > 8_000_000:
                raise ServiceCheckError("Voice catalogue exceeded the safe response size limit.")
            data = json.loads(raw.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            status = exc.code
            exc.close()
            hint = {
                401: "check the API key",
                403: "enable Voices Read permission",
                429: "check rate limits",
            }.get(status, "check ElevenLabs availability")
            raise ServiceCheckError(f"ElevenLabs HTTP {status}: {hint}.") from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise ServiceCheckError("ElevenLabs connection failed; check network access.") from None
        except (ValueError, UnicodeError):
            raise ServiceCheckError("ElevenLabs returned invalid JSON.") from None
        if not isinstance(data, dict) or not isinstance(data.get("voices"), list):
            raise ServiceCheckError("ElevenLabs returned an unexpected voice catalogue format.")
        for voice in data["voices"]:
            if not isinstance(voice, dict) or not isinstance(voice.get("voice_id"), str):
                raise ServiceCheckError("ElevenLabs returned an invalid voice record.")
            # Print only the selection fields; discard samples, recordings, URLs, etc.
            labels = voice.get("labels", {})
            voices[voice["voice_id"]] = {
                "voice_id": voice["voice_id"],
                "name": voice.get("name", ""),
                "labels": labels if isinstance(labels, dict) else {},
            }
        if not data.get("has_more", False):
            break
        token = data.get("next_page_token")
        if not isinstance(token, str) or not token or token in seen_tokens:
            raise ServiceCheckError("ElevenLabs returned invalid pagination; catalogue may be incomplete.")
        seen_tokens.add(token)
    else:
        raise ServiceCheckError("Voice catalogue exceeded 20 pages; narrow the catalogue in ElevenLabs.")
    result = sorted(voices.values(), key=lambda voice: (str(voice["name"]).lower(), voice["voice_id"]))
    print("Available ElevenLabs voices (existing IDs; this command does not create or clone voices):")
    for voice in result:
        print(_safe_text(json.dumps(voice, ensure_ascii=False, sort_keys=True)))
    print(f"Voices returned: {len(result)}")
    print("Copy selected IDs into ELEVENLABS_VOICE_VERA / MARK / ECHO / NARRATOR in .env.")
    return result


def check_audio() -> None:
    play = {
        "title": "Проверка озвучки",
        "metadata": {"mode": "audio_smoke", "full_play_completed": False},
        "turns": [
            {
                "speaker": "vera",
                "act": 1,
                "stage_direction": "В читальном зале загорается последняя лампа.",
                "text": "Добро пожаловать в театр. Эта реплика проверяет голос Веры.",
            }
        ],
    }
    paths = render_audio(play, ROOT / "outputs" / "audio_smoke")
    print("Audio check: OK; real narration and Vera speech synthesized.")
    print(f"Manifest: {paths['manifest']}")
    print(f"Audio: {paths['audio'] or paths['playlist']}")
    print("This is a short service check, not the final play recording.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--cloud", action="store_true", help="Make one small real cloud LLM request (may use paid quota)"
    )
    modes.add_argument(
        "--voices", action="store_true", help="Read existing ElevenLabs voice IDs and selection labels"
    )
    modes.add_argument(
        "--audio-smoke",
        action="store_true",
        help="Synthesize two short clips (Vera + narrator; may use paid quota)",
    )
    args = parser.parse_args(argv)
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("Install project dependencies first: uv sync --extra notebook --extra dev", file=sys.stderr)
        return 2
    load_dotenv(ROOT / ".env", override=False)
    try:
        if args.cloud:
            check_cloud()
        elif args.voices:
            list_voices()
        elif args.audio_smoke:
            check_audio()
        else:
            offline_report()
    except (ProviderError, AudioError, ServiceCheckError) as exc:
        print("Service check failed: " + _safe_text(exc), file=sys.stderr)
        return 1
    except (ValueError, TypeError, OSError):
        # Invalid numeric settings, unreadable local paths, etc. can embed user
        # input in exception text, so report a fixed diagnostic instead.
        print(
            "Service check failed: invalid configuration or inaccessible local output path. Check .env and the setup instructions.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
