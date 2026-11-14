"""Service preflight stays offline by default and never prints credentials."""

import io
import json
import os
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

from scripts import check_services


class ServiceCheckTests(unittest.TestCase):
    def test_offline_report_never_calls_network_or_prints_secrets(self):
        secret = "test-hidden-credential-123456"
        with (
            patch.dict(os.environ, {"LLM_API_KEY": secret, "ELEVENLABS_API_KEY": secret}, clear=True),
            patch("scripts.check_services.urllib.request.urlopen") as network,
            patch("scripts.check_services.importlib.util.find_spec", return_value=None),
            redirect_stdout(io.StringIO()) as output,
        ):
            check_services.offline_report()
        self.assertNotIn(secret, output.getvalue())
        self.assertIn("Cloud API key (LLM_API_KEY / YANDEX_API_KEY / api_key): SET", output.getvalue())
        self.assertIn("CUDA: NOT PROBED", output.getvalue())
        network.assert_not_called()

    def test_cloud_smoke_one_small_request_without_echoing_response(self):
        provider = MagicMock()
        provider.timeout = 120
        provider.complete.return_value = "untrusted-service-output"
        with (
            patch("scripts.check_services.CloudProvider.from_env", return_value=provider),
            redirect_stdout(io.StringIO()) as output,
        ):
            check_services.check_cloud()
        self.assertEqual(provider.max_attempts, 1)
        self.assertEqual(provider.timeout, 60)
        self.assertEqual(provider.complete.call_args.kwargs["max_tokens"], 100)
        self.assertNotIn("untrusted-service-output", output.getvalue())

    def test_voice_pagination_discards_samples_and_redacts_key(self):
        secret = "test-hidden-credential-123456"
        first = {
            "voices": [
                {
                    "voice_id": "v1",
                    "name": "Vera",
                    "labels": {"language": "ru"},
                    "samples": [{"private": "sample-data"}],
                }
            ],
            "has_more": True,
            "next_page_token": "cursor-2",
        }
        second = {
            "voices": [
                {
                    "voice_id": "v2",
                    "name": "Narrator " + secret,
                    "labels": {"gender": "male"},
                    "preview_url": "private-link",
                }
            ],
            "has_more": False,
        }
        responses = []
        for page in (first, second):
            response = MagicMock()
            response.__enter__.return_value = response
            response.read.return_value = json.dumps(page).encode()
            responses.append(response)
        with (
            patch.dict(os.environ, {"ELEVENLABS_API_KEY": secret}, clear=True),
            patch("scripts.check_services.urllib.request.urlopen", side_effect=responses) as transport,
            redirect_stdout(io.StringIO()) as output,
        ):
            voices = check_services.list_voices()
        self.assertEqual(len(voices), 2)
        self.assertEqual(transport.call_count, 2)
        self.assertIn("/v2/voices?", transport.call_args_list[0].args[0].full_url)
        self.assertIn("next_page_token=cursor-2", transport.call_args_list[1].args[0].full_url)
        for value in (secret, "sample-data", "preview_url", "private-link"):
            self.assertNotIn(value, output.getvalue())
        self.assertIn("[REDACTED]", output.getvalue())


if __name__ == "__main__":
    unittest.main()
