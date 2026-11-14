"""Multivoice rendering through ElevenLabs or local Piper on CPU.

API source: https://elevenlabs.io/docs/api-reference/text-to-speech/convert
No voice cloning is used. Select existing voice IDs from your ElevenLabs account.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import wave
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


class AudioError(RuntimeError):
    pass


PIPER_VOICES = {
    "vera": "ru_RU-irina-medium",
    "mark": "ru_RU-dmitri-medium",
    "echo": "ru_RU-ruslan-medium",
    "narrator": "ru_RU-denis-medium",
}
PIPER_PACING = {"vera": 1.04, "mark": 0.94, "echo": 1.12, "narrator": 1.08}


def _audio_lines(result: dict, include_directions: bool = True) -> list[dict]:
    lines: list[dict] = []
    scenes = {
        scene["act"]: scene
        for scene in result.get("scenes", [])
        if isinstance(scene, dict) and "act" in scene
    }
    seen_acts = set()
    for turn in result.get("turns", []):
        act = turn.get("act", 1)
        if act not in seen_acts:
            scene = scenes.get(act, {})
            scene_text = scene.get("text", scene.get("scene", ""))
            if isinstance(scene_text, str) and scene_text.strip():
                lines.append({"speaker": "narrator", "text": scene_text.strip(), "act": act, "kind": "scene"})
            seen_acts.add(act)
        direction = turn.get("stage_direction", "")
        if include_directions and isinstance(direction, str) and direction.strip():
            lines.append({"speaker": "narrator", "text": direction.strip(), "act": act, "kind": "direction"})
        text = turn.get("text", "")
        speaker = turn.get("speaker", "")
        if (
            not isinstance(text, str)
            or not text.strip()
            or not isinstance(speaker, str)
            or not speaker.strip()
        ):
            raise AudioError("Each turn needs non-empty speaker and text")
        kind = "scene" if speaker == "narrator" else "speech"
        lines.append({"speaker": speaker, "text": text.strip(), "act": act, "kind": kind})
    if not lines:
        raise AudioError("The play has no lines to synthesize")
    if any(len(line["text"]) > 4500 for line in lines):
        raise AudioError("An audio line exceeds 4500 characters; split it before synthesis")
    return lines


def render_local_audio(
    result: dict,
    output_dir: str | Path,
    *,
    model_dir: str | Path,
    include_directions: bool = True,
) -> dict:
    """Render the actual play to one WAV with an inspectable cue list.

    Piper runs on CPU and reads previously downloaded models. Synthesis sends
    no text over the network. The built-in wave module assembles the result.
    """
    lines = _audio_lines(result, include_directions)
    speakers = sorted({line["speaker"] for line in lines})
    unknown = sorted(set(speakers) - PIPER_VOICES.keys())
    if unknown:
        raise AudioError("No Piper voice assigned for: " + ", ".join(unknown))

    models = {speaker: Path(model_dir) / f"{PIPER_VOICES[speaker]}.onnx" for speaker in speakers}
    missing = [
        path.name
        for model in models.values()
        for path in (model, model.with_suffix(".onnx.json"))
        if not path.is_file()
    ]
    if missing:
        raise AudioError("Download missing Piper voice files: " + ", ".join(missing))
    try:
        from piper import PiperVoice, SynthesisConfig

        piper_version = version("piper-tts")
    except (ImportError, PackageNotFoundError) as error:
        raise AudioError("Install Piper with python -m pip install piper-tts") from error

    voices = {speaker: PiperVoice.load(path) for speaker, path in models.items()}
    pacing = {
        speaker: SynthesisConfig(length_scale=PIPER_PACING[speaker], volume=0.8)
        for speaker in speakers
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    audio_path = out / "play.wav"
    manifest_path = out / "manifest.json"
    clips = []
    audio_format = None
    frame_offset = 0
    with tempfile.TemporaryDirectory(prefix=".piper-", dir=out) as temporary:
        temp_audio = Path(temporary) / audio_path.name
        temp_manifest = Path(temporary) / manifest_path.name
        with wave.open(str(temp_audio), "wb") as recording:
            for index, line in enumerate(lines, 1):
                speaker = line["speaker"]
                buffer = io.BytesIO()
                with wave.open(buffer, "wb") as clip:
                    voices[speaker].synthesize_wav(line["text"], clip, syn_config=pacing[speaker])
                buffer.seek(0)
                with wave.open(buffer, "rb") as clip:
                    clip_format = (clip.getnchannels(), clip.getsampwidth(), clip.getframerate())
                    if audio_format is None:
                        audio_format = clip_format
                        recording.setnchannels(clip_format[0])
                        recording.setsampwidth(clip_format[1])
                        recording.setframerate(clip_format[2])
                    if clip_format != audio_format or clip_format[:2] != (1, 2):
                        raise AudioError("Piper voices must share the same mono 16-bit PCM format")
                    frames = clip.getnframes()
                    if frames <= 0:
                        raise AudioError(f"Piper returned an empty clip at line {index}")
                    if clips:
                        gap = 1.15 if line["act"] != clips[-1]["act"] else (
                            0.22 if clips[-1]["kind"] == "direction" else 0.45
                        )
                        silence_frames = round(clip_format[2] * gap)
                        recording.writeframesraw(b"\x00\x00" * silence_frames)
                        frame_offset += silence_frames
                    recording.writeframesraw(clip.readframes(frames))
                clips.append(
                    {
                        **line,
                        "index": index,
                        "voice_id": PIPER_VOICES[speaker],
                        "start_seconds": round(frame_offset / clip_format[2], 3),
                        "end_seconds": round((frame_offset + frames) / clip_format[2], 3),
                    }
                )
                frame_offset += frames
        manifest = {
            "status": "complete",
            "provider": "Piper",
            "provider_version": piper_version,
            "device": "cpu",
            "source_mode": result.get("metadata", {}).get("mode", result.get("mode", "unknown")),
            "title": result.get("title", ""),
            "voices": {speaker: PIPER_VOICES[speaker] for speaker in speakers},
            "length_scale": {speaker: PIPER_PACING[speaker] for speaker in speakers},
            "sample_rate": audio_format[2],
            "merged": audio_path.name,
            "duration_seconds": round(frame_offset / audio_format[2], 3),
            "clips": clips,
        }
        temp_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_audio.replace(audio_path)
        temp_manifest.replace(manifest_path)
    return {
        "manifest": str(manifest_path),
        "playlist": None,
        "audio": str(audio_path),
        "clips": len(clips),
        "characters": sum(len(line["text"]) for line in lines),
    }


def render_audio(
    result: dict,
    output_dir: str | Path,
    *,
    voices: dict[str, str] | None = None,
    api_key: str | None = None,
    model_id: str = "eleven_multilingual_v2",
    include_directions: bool = True,
    merge: bool = True,
    timeout: float = 90,
    base_url: str = "https://api.elevenlabs.io/v1",
) -> dict:
    """Synthesize the actual play, one fixed voice per speaker, resuming cached lines.

    Configure ELEVENLABS_API_KEY and ELEVENLABS_VOICE_VERA / MARK / ECHO /
    NARRATOR, or pass explicit voices. Missing credentials never create fake audio.
    Per-line files, JSON manifest and M3U playlist remain useful without FFmpeg.
    A failed request preserves completed lines; rerunning resumes by content hash.
    """
    api_key = api_key or os.getenv("ELEVENLABS_API_KEY", "")
    if not api_key or api_key.strip() in {"...", "your-key", "YOUR_API_KEY"}:
        raise AudioError("Set ELEVENLABS_API_KEY in the local .env before synthesis")
    lines = _audio_lines(result, include_directions)
    speakers = {line["speaker"] for line in lines}
    selected = dict(voices or {})
    for speaker in speakers:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,40}", speaker):
            raise AudioError("Use stable speaker IDs such as vera, mark, echo, narrator")
        selected.setdefault(speaker, os.getenv(f"ELEVENLABS_VOICE_{speaker.upper()}", ""))
    missing = sorted(speaker for speaker in speakers if not selected[speaker] or selected[speaker] == "...")
    if missing:
        raise AudioError("Configure existing ElevenLabs voice IDs for: " + ", ".join(missing))
    if len({selected[speaker] for speaker in speakers}) != len(speakers):
        raise AudioError("Choose a distinct existing voice ID for each speaker")
    if not 1 <= timeout <= 300:
        raise AudioError("timeout must be between 1 and 300 seconds")
    parsed_url = urllib.parse.urlparse(base_url)
    if parsed_url.scheme != "https" and parsed_url.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise AudioError("Audio API requires HTTPS (HTTP is allowed only for local tests)")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    previous = {}
    if manifest_path.exists():
        try:
            previous = {
                item["cache_key"]: item
                for item in json.loads(manifest_path.read_text(encoding="utf-8")).get("clips", [])
            }
        except (ValueError, KeyError, TypeError):
            previous = {}
    manifest = {
        "status": "in_progress",
        "provider": "ElevenLabs",
        "model": model_id,
        "source_mode": result.get("metadata", {}).get("mode", result.get("mode", "unknown")),
        "title": result.get("title", ""),
        "voices": {speaker: selected[speaker] for speaker in sorted(speakers)},
        "clips": [],
        "merged": None,
    }

    def persist() -> None:
        temp = manifest_path.with_suffix(".tmp")
        temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(manifest_path)

    persist()
    for index, line in enumerate(lines, 1):
        voice = selected[line["speaker"]]
        payload = {
            "text": line["text"],
            "model_id": model_id,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        }
        # Context differs by position, so it participates in the cache identity.
        if index > 1:
            payload["previous_text"] = lines[index - 2]["text"]
        if index < len(lines):
            payload["next_text"] = lines[index]["text"]
        identity = json.dumps({"voice": voice, "payload": payload}, sort_keys=True, ensure_ascii=False)
        cache_key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        filename = f"{index:04d}_{line['speaker']}_{cache_key[:10]}.mp3"
        target = out / filename
        cached = previous.get(cache_key, {})
        request_id = cached.get("request_id")
        # Only trust the exact generated filename inside our output directory.
        cached_file = out / Path(str(cached.get("file", ""))).name
        reusable = (
            cached.get("file")
            and cached_file.is_file()
            and cached_file.stat().st_size > 0
            and cached_file.suffix == ".mp3"
            and not cached_file.is_symlink()
        )
        if reusable:
            reusable = hashlib.sha256(cached_file.read_bytes()).hexdigest() == cached.get("sha256")
        if reusable:
            if cached_file != target:
                shutil.copyfile(cached_file, target)
            reused = True
        else:
            url = (
                base_url.rstrip("/")
                + "/text-to-speech/"
                + urllib.parse.quote(voice, safe="")
                + "?output_format=mp3_44100_128"
            )
            request = urllib.request.Request(
                url,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"xi-api-key": api_key, "Content-Type": "application/json", "Accept": "audio/mpeg"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    content_type = response.headers.get("Content-Type", "").split(";")[0].lower()
                    if content_type not in {"audio/mpeg", "audio/mp3", "application/octet-stream"}:
                        raise AudioError(f"Audio API returned unexpected content type: {content_type}")
                    audio = response.read(30_000_001)
                    request_id = response.headers.get("request-id") or response.headers.get("x-request-id")
                if not audio or len(audio) > 30_000_000:
                    raise AudioError("Audio API returned an empty or oversized clip")
                temp = target.with_suffix(".tmp")
                temp.write_bytes(audio)
                temp.replace(target)
                reused = False
            except urllib.error.HTTPError as exc:
                manifest["status"] = "partial"
                persist()
                # Never print the response payload or headers: they can contain credentials.
                raise AudioError(
                    f"ElevenLabs HTTP {exc.code} at clip {index}; completed clips retained. Check key, voice access and quota."
                ) from None
            except (urllib.error.URLError, TimeoutError, OSError, AudioError) as exc:
                manifest["status"] = "partial"
                persist()
                raise AudioError(
                    f"Audio generation failed at clip {index} ({type(exc).__name__}); completed clips retained"
                ) from None
        manifest["clips"].append(
            {
                **line,
                "index": index,
                "file": filename,
                "voice_id": voice,
                "cache_key": cache_key,
                "request_id": request_id,
                "reused": reused,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        )
        persist()
    playlist = out / "play.m3u"
    playlist.write_text(
        "#EXTM3U\n" + "\n".join(clip["file"] for clip in manifest["clips"]) + "\n", encoding="utf-8"
    )
    ffmpeg = shutil.which("ffmpeg")
    if merge and ffmpeg:
        concat = out / "concat.txt"
        # Filenames are generated here, so the concat list contains no untrusted paths.
        concat.write_text(
            "\n".join(f"file '{clip['file']}'" for clip in manifest["clips"]) + "\n", encoding="utf-8"
        )
        merged = out / "play.mp3"
        try:
            process = subprocess.run(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "1",
                    "-i",
                    str(concat.resolve()),
                    "-c",
                    "copy",
                    str(merged.resolve()),
                ],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            merged_ok = process.returncode == 0 and merged.exists() and merged.stat().st_size > 0
        except (OSError, subprocess.TimeoutExpired):
            merged_ok = False
        if merged_ok:
            manifest["merged"] = merged.name
        else:
            manifest["merge_warning"] = (
                "FFmpeg could not merge clips; individual MP3 files and playlist are available"
            )
    elif merge:
        manifest["merge_warning"] = "FFmpeg not found; individual MP3 files and playlist are available"
    manifest["status"] = "complete"
    manifest["playlist"] = playlist.name
    persist()
    return {
        "manifest": str(manifest_path),
        "playlist": str(playlist),
        "audio": str(out / manifest["merged"]) if manifest["merged"] else None,
        "clips": len(manifest["clips"]),
        "characters": sum(len(line["text"]) for line in lines),
    }
