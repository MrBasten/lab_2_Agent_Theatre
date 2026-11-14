r"""Run a real local Hugging Face role response and save reproducible evidence.

Run from the project root: .venv\Scripts\python scripts\check_local_model.py
This verifies one local model response; it does not claim a complete play run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from theatre.providers import HFLocalProvider, ProviderError  # noqa: E402

MESSAGES = [
    {
        "role": "system",
        "content": (
            "Ты Эхо, локальный ИИ-хранитель городского архива в пьесе. "
            "Ты ценишь память о людях, говоришь сдержанно и образно. "
            'Ответь по-русски, только JSON-объектом вида {"text":"твоя реплика"}. '
            "Одна реплика, 1-2 предложения. Не добавляй рассуждения и Markdown."
        ),
    },
    {
        "role": "user",
        "content": (
            "Марк: На содержание архива больше нет электричества. "
            "Я предлагаю стереть личные письма и оставить только таблицы. "
            "Эхо, что ты возразишь?"
        ),
    },
]


def installed_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def inspect_answer(raw: str) -> tuple[str, dict]:
    """Check observable properties; literary quality still needs a human review."""
    content = raw.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content).strip()
    try:
        parsed = json.loads(content)
    except (ValueError, TypeError):
        parsed = None
    text = parsed.get("text", "") if isinstance(parsed, dict) else ""
    text = text.strip() if isinstance(text, str) else ""
    checks = {
        "valid_json_object": isinstance(parsed, dict),
        "nonempty_role_text": len(text) >= 20,
        "russian_text": len(re.findall(r"[А-Яа-яЁё]", text)) >= 10,
        "no_thinking_block": "<think>" not in raw and "</think>" not in raw,
    }
    checks["passed"] = all(checks.values())
    return text, checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "local_smoke_latest.json")
    parser.add_argument("--max-tokens", type=int, default=220)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    try:
        import torch
        from dotenv import load_dotenv
    except ImportError:
        print("Local smoke requires the project local extra and python-dotenv.", file=sys.stderr)
        return 2
    load_dotenv(ROOT / ".env", override=False)
    provider = HFLocalProvider(
        model_id=os.getenv("LOCAL_MODEL_ID", "Qwen/Qwen3-4B-Instruct-2507"),
        device=os.getenv("LOCAL_DEVICE", "auto"),
        seed=arguments.seed,
        revision=os.getenv("LOCAL_MODEL_REVISION") or None,
        quantization=os.getenv("LOCAL_QUANTIZATION") or "none",
    )
    cuda_available = torch.cuda.is_available()
    record = {
        "task": "single local Russian character response; not a complete play",
        "provider": provider.name,
        "real_inference": False,
        "full_play_completed": False,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "seed": arguments.seed,
        "temperature": 0.7,
        "max_tokens": arguments.max_tokens,
        "requested_device": provider.device,
        "quantization": provider.quantization,
        "cuda_available": cuda_available,
        "cuda_device": torch.cuda.get_device_name(0) if cuda_available else None,
        "cuda_total_memory_mib": (
            round(torch.cuda.get_device_properties(0).total_memory / 1024**2, 2) if cuda_available else None
        ),
        "cuda_runtime": torch.version.cuda,
        "versions": {
            package: installed_version(package)
            for package in (
                "torch",
                "transformers",
                "accelerate",
                "safetensors",
                "bitsandbytes",
                "python-dotenv",
            )
        },
        "python_version": sys.version,
        "messages": MESSAGES,
    }
    started = time.monotonic()
    exit_code = 0
    if cuda_available:
        torch.cuda.reset_peak_memory_stats(0)
    try:
        raw = provider.complete(MESSAGES, temperature=0.7, max_tokens=arguments.max_tokens)
        role_text, checks = inspect_answer(raw)
        record.update(real_inference=True, output=raw, role_text=role_text, checks=checks)
        record["actual_device"] = provider.calls[-1]["device"]
        record["resolved_revision"] = provider.resolved_revision
        record["hf_device_map"] = {
            layer: str(device) for layer, device in getattr(provider._model, "hf_device_map", {}).items()
        }
        # Some Transformers versions omit hf_device_map even after placement.
        # Inspect actual tensors as independent evidence of CPU/meta offloading.
        parameter_bytes_by_device: dict[str, int] = {}
        for parameter in provider._model.parameters():
            device = str(parameter.device)
            parameter_bytes_by_device[device] = (
                parameter_bytes_by_device.get(device, 0) + parameter.numel() * parameter.element_size()
            )
        record["parameter_devices"] = sorted(parameter_bytes_by_device)
        record["parameter_memory_mib_by_device"] = {
            device: round(size / 1024**2, 2) for device, size in parameter_bytes_by_device.items()
        }
        if not checks["passed"]:
            exit_code = 1
    except ProviderError as error:
        record.update(error=str(error), checks={"passed": False})
        exit_code = 1
    record["wall_seconds"] = round(time.monotonic() - started, 3)
    record["calls"] = provider.calls
    record["cuda_peak_memory_allocated_mib"] = (
        round(torch.cuda.max_memory_allocated(0) / 1024**2, 2) if cuda_available else None
    )
    record["cuda_peak_memory_reserved_mib"] = (
        round(torch.cuda.max_memory_reserved(0) / 1024**2, 2) if cuda_available else None
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Provider: {provider.name}")
    print(f"Passed: {record['checks']['passed']}; wall seconds: {record['wall_seconds']}")
    print(record.get("role_text") or record.get("output") or record.get("error"))
    print(f"Evidence: {arguments.output}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
