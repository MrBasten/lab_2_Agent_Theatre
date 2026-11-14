"""Tool security, numerical evidence and real HTTP adapter contracts, all offline."""

import json
import os
import tempfile
import threading
import unittest
import urllib.error
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from theatre.audio import PIPER_VOICES, AudioError, render_audio, render_local_audio
from theatre.tools import ToolBox

ROOT = Path(__file__).resolve().parents[1]


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.tools = ToolBox(ROOT / "data" / "archive")

    def test_search_then_read_and_cost_resolve_plot_evidence(self):
        hits = self.tools.execute("search_files", {"query": "смета волонтёры"})["hits"]
        self.assertEqual(hits[0]["id"], "D2")
        doc = self.tools.execute("read_file", {"id": hits[0]["id"]})
        self.assertIn("1200", doc["text"])
        self.assertTrue(doc["fictional"])
        cost = self.tools.execute(
            "estimate_preservation_cost", {"pages": 1200, "cost_per_page": 2.5, "volunteers": 12}
        )
        self.assertEqual(cost["total_rub"], 1800.0)
        self.assertEqual(cost["paid_pages"], 720)

    def test_search_bounds_unknown_tools_and_bad_numbers(self):
        for args in (
            {"query": ""},
            {"query": "x" * 401},
            {"query": "архив", "limit": 100},
            {"query": "архив", "limit": True},
        ):
            self.assertIn("error", self.tools.execute("search_files", args))
        self.assertIn("error", self.tools.execute("execute_python", {"code": "anything"}))
        for args in (
            {"pages": -1},
            {"pages": True},
            {"pages": 1, "cost_per_page": float("nan")},
            {"pages": 1, "cost_per_page": float("inf")},
            {"pages": 1, "volunteers": -2},
        ):
            self.assertIn("error", self.tools.execute("estimate_preservation_cost", args))

    def test_paths_cannot_escape_archive(self):
        for value in (
            "../README.md",
            "..\\README.md",
            str(ROOT / "README.md"),
            "C:README.md",
            "D1_order.md:secret",
            "README.md",
        ):
            self.assertIn("error", self.tools.execute("read_file", {"path": value}), value)
        self.assertEqual(self.tools.execute("read_file", {"path": "D1_order.md"})["id"], "D1")

    def test_redundant_source_address_must_identify_same_document(self):
        document = self.tools.execute("read_file", {"id": "D4", "path": "D4_log.md"})
        self.assertEqual(document, self.tools.execute("read_file", {"id": "D4"}))
        for path in ("D1_order.md", "../README.md", {"path": "D4_log.md"}):
            with self.subTest(path=path):
                self.assertIn("error", self.tools.execute("read_file", {"id": "D4", "path": path}))

    def test_empty_search_does_not_invent_sources(self):
        self.assertEqual(self.tools.execute("search_files", {"query": "xyzzyplugh"})["hits"], [])

    def test_duplicate_source_id_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "D1_a.md").write_text("one")
            Path(directory, "D1_b.md").write_text("two")
            with self.assertRaises(ValueError):
                ToolBox(directory)


class AudioTests(unittest.TestCase):
    def test_local_audio_routes_scene_remark_and_dialogue_to_timed_voices(self):
        play = {
            "metadata": {"mode": "cloud"},
            "turns": [
                {"act": 1, "speaker": "narrator", "text": "Сцена."},
                {"act": 1, "speaker": "vera", "stage_direction": "Шепчет.", "text": "Свет."},
                {"act": 2, "speaker": "mark", "text": "Пуск."},
                {"act": 2, "speaker": "echo", "text": "Готово."},
            ],
        }
        calls = []

        class FakeVoice:
            def __init__(self, path):
                self.path = path

            def synthesize_wav(self, text, stream, *, syn_config):
                calls.append((self.path.stem, text, syn_config.length_scale))
                stream.setparams((1, 2, 1000, 0, "NONE", "not compressed"))
                stream.writeframes(b"\x01\x00" * 100)

        module = SimpleNamespace(
            PiperVoice=SimpleNamespace(load=FakeVoice), SynthesisConfig=SimpleNamespace
        )
        with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules", {"piper": module}), patch(
            "theatre.audio.version", return_value="test"
        ):
            models = Path(directory, "models")
            models.mkdir()
            for voice in PIPER_VOICES.values():
                for suffix in (".onnx", ".onnx.json"):
                    (models / (voice + suffix)).touch()
            paths = render_local_audio(play, Path(directory, "audio"), model_dir=models)
            self.assertEqual(paths["clips"], 5)
            manifest = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["source_mode"], "cloud")
            self.assertEqual(manifest["voices"], PIPER_VOICES)
            self.assertEqual([clip["kind"] for clip in manifest["clips"]], [
                "scene", "direction", "speech", "speech", "speech"
            ])
            self.assertEqual([clip["start_seconds"] for clip in manifest["clips"]], [
                0.0, 0.55, 0.87, 2.12, 2.67
            ])
            self.assertEqual([voice for voice, _, _ in calls], [
                PIPER_VOICES[role] for role in ("narrator", "narrator", "vera", "mark", "echo")
            ])
            with wave.open(paths["audio"], "rb") as stream:
                self.assertEqual(stream.getnframes(), 2770)

    def test_missing_local_voice_fails_without_writing_audio(self):
        play = {"turns": [{"speaker": "vera", "text": "Каталог нужно сохранить."}]}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(AudioError, "ru_RU-irina-medium.onnx"):
                render_local_audio(play, directory, model_dir=directory)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_missing_configuration_never_writes_fake_audio(self):
        play = {"turns": [{"speaker": "vera", "text": "Каталог нужно сохранить."}]}
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(AudioError):
                render_audio(play, directory)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_distinct_voices_preflight(self):
        play = {"turns": [{"speaker": "vera", "text": "Да."}, {"speaker": "mark", "text": "Нет."}]}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(AudioError, "distinct"):
                render_audio(
                    play, directory, api_key="fake-test-key", voices={"vera": "same", "mark": "same"}
                )

    def test_failed_request_keeps_completed_clips_and_resumes(self):
        play = {
            "turns": [
                {"speaker": "vera", "text": "Первая реплика."},
                {"speaker": "mark", "text": "Вторая реплика."},
            ]
        }
        response = MagicMock()
        response.__enter__.return_value = response
        response.headers = {"Content-Type": "audio/mpeg", "request-id": "offline"}
        response.read.return_value = b"ID3-offline-test"
        failure = urllib.error.HTTPError("https://api.elevenlabs.io", 429, "rate limit", {}, None)
        kwargs = {"api_key": "fake-test-key", "voices": {"vera": "voice1", "mark": "voice2"}, "merge": False}
        with tempfile.TemporaryDirectory() as directory:
            with patch("theatre.audio.urllib.request.urlopen", side_effect=[response, failure]):
                with self.assertRaisesRegex(AudioError, "HTTP 429"):
                    render_audio(play, directory, **kwargs)
            manifest = json.loads(Path(directory, "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "partial")
            self.assertEqual(len(manifest["clips"]), 1)
            with patch("theatre.audio.urllib.request.urlopen", return_value=response) as transport:
                render_audio(play, directory, **kwargs)
                self.assertEqual(transport.call_count, 1)
            # A damaged cache entry is regenerated instead of silently reused.
            first_clip = Path(directory, manifest["clips"][0]["file"])
            first_clip.write_bytes(b"damaged")
            with patch("theatre.audio.urllib.request.urlopen", return_value=response) as transport:
                render_audio(play, directory, **kwargs)
                self.assertEqual(transport.call_count, 1)

    def test_http_protocol_narration_and_resumable_cache(self):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append(
                    {"path": self.path, "key": self.headers.get("xi-api-key"), "payload": payload}
                )
                self.send_response(200)
                self.send_header("Content-Type", "audio/mpeg")
                self.send_header("request-id", "test-request")
                self.end_headers()
                # Test bytes exercise the transport only, never a playable artefact.
                self.wfile.write(b"ID3-offline-transport-test")

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            play = {
                "title": "test",
                "metadata": {"mode": "demo"},
                "scenes": [{"act": 1, "text": "Лампа мигает."}],
                "turns": [
                    {
                        "speaker": "vera",
                        "text": "Сохраним каталог.",
                        "stage_direction": "Открывает ящик.",
                        "act": 1,
                    }
                ],
            }
            kwargs = {
                "api_key": "fake-test-key",
                "voices": {"vera": "voice-vera", "narrator": "voice-narrator"},
                "base_url": f"http://127.0.0.1:{server.server_port}/v1",
                "merge": False,
            }
            with tempfile.TemporaryDirectory() as directory:
                result = render_audio(play, directory, **kwargs)
                self.assertEqual(result["clips"], 3)
                self.assertEqual(len(requests), 3)
                self.assertIn("voice-narrator", requests[0]["path"])
                self.assertEqual(requests[2]["payload"]["text"], "Сохраним каталог.")
                self.assertEqual(requests[2]["key"], "fake-test-key")
                self.assertEqual(requests[0]["payload"]["model_id"], "eleven_multilingual_v2")
                manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
                self.assertEqual(manifest["status"], "complete")
                self.assertEqual(manifest["source_mode"], "demo")
                self.assertNotIn("fake-test-key", Path(result["manifest"]).read_text(encoding="utf-8"))
                render_audio(play, directory, **kwargs)
                self.assertEqual(len(requests), 3)
                play["turns"][0]["text"] = "Сохраним каталог иначе."
                render_audio(play, directory, **kwargs)
                self.assertEqual(len(requests), 5)  # Changed line + previous line's next_text.
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
