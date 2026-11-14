"""CLI: python -m theatre --mode demo|cloud|live|local [--audio]."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from .engine import demo_providers, run_play, save_result
from .providers import CloudProvider, HFLocalProvider
from .tools import ToolBox

ROOT = Path(__file__).resolve().parents[1]


def build_providers(mode: str, seed: int = 42) -> dict:
    """Only an explicit demo choice uses fixtures; live errors stay errors."""
    if mode == "demo":
        return demo_providers()
    if mode in {"live", "local"}:
        local = HFLocalProvider(
            model_id=os.getenv("LOCAL_MODEL_ID") or "Qwen/Qwen3-4B-Instruct-2507",
            device=os.getenv("LOCAL_DEVICE") or "auto",
            seed=seed,
            revision=os.getenv("LOCAL_MODEL_REVISION") or None,
            quantization=os.getenv("LOCAL_QUANTIZATION") or "none",
        )
    if mode == "local":
        return {key: local for key in ("planner", "director", "narrator", "vera", "mark", "echo")}
    if mode not in {"cloud", "live"}:
        raise ValueError(f"Unknown theatre mode: {mode}")
    cloud = CloudProvider.from_env()
    mark_model = os.getenv("MARK_MODEL", "").strip()
    mark = CloudProvider.from_env(model=mark_model) if mark_model else cloud
    return {
        "planner": cloud,
        "director": cloud,
        "narrator": cloud,
        "vera": cloud,
        "mark": mark,
        "echo": local if mode == "live" else cloud,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Архив последнего света — агентский театр")
    parser.add_argument("--mode", choices=("demo", "cloud", "live", "local"), default="demo")
    parser.add_argument("--acts", type=int, default=3)
    parser.add_argument("--turns", type=int, default=6, help="Реплики персонажей в каждом действии")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audio", action="store_true", help="Озвучить через настроенный ElevenLabs")
    parser.add_argument(
        "--audio-from", type=Path, help="Озвучить сохранённый play.json без повторной генерации"
    )
    parser.add_argument("--piper-model-dir", type=Path, help="Озвучка через Piper с локальными голосами")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env", override=False)

    def synthesize(result, output):
        if args.piper_model_dir is not None:
            from .audio import render_local_audio

            return render_local_audio(result, output, model_dir=args.piper_model_dir)
        from .audio import render_audio

        return render_audio(result, output)

    if args.audio_from:
        saved = json.loads(args.audio_from.read_text(encoding="utf-8"))
        audio = synthesize(saved, args.audio_from.parent / ("audio_piper" if args.piper_model_dir else "audio"))
        print(json.dumps(audio, ensure_ascii=False, indent=2, default=str))
        return
    # Retain the published 1.7B baseline in outputs/local for comparison.
    default_folder = "local_4b" if args.mode == "local" else args.mode
    output = args.output or ROOT / "outputs" / default_folder
    providers = build_providers(args.mode, args.seed)
    print(f"Режим: {args.mode}; действий: {args.acts}; реплик/действие: {args.turns}", flush=True)
    print("Модели: " + ", ".join(f"{key}={provider.name}" for key, provider in providers.items()), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    # Keep an incremental trace even if a real model fails midway through a play.
    with (output / "events.jsonl").open("w", encoding="utf-8") as trace:

        def on_event(event):
            trace.write(json.dumps(event, ensure_ascii=False) + "\n")
            trace.flush()
            if event["type"] == "selection":
                print(f"Действие {event['act']}: {event['speaker']} ({event['guard']})", flush=True)

        result = run_play(
            providers,
            tools=ToolBox(ROOT / "data" / "archive"),
            turns_per_act=args.turns,
            acts=args.acts,
            mode=args.mode,
            seed=args.seed,
            on_event=on_event,
        )
    paths = save_result(result, output)
    print(json.dumps(paths, ensure_ascii=False, indent=2, default=str))
    if args.audio:
        audio = synthesize(result, output / ("audio_piper" if args.piper_model_dir else "audio"))
        print(json.dumps(audio, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
