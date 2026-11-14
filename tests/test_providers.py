"""Offline adapter checks: a loopback HTTP server and a tiny fake tensor model."""

import json
import threading
import unittest
from contextlib import contextmanager, nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

from theatre.providers import CloudProvider, HFLocalProvider, ProviderError, _retry_delay


@contextmanager
def api_server(responses):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(
                {
                    "path": self.path,
                    "headers": dict(self.headers),
                    "body": json.loads(self.rfile.read(int(self.headers["Content-Length"]))),
                }
            )
            status, body = responses.pop(0)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Retry-After", "0")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


CHAT = {
    "choices": [{"message": {"content": "Ответ"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 12, "completion_tokens": 3},
}
MESSAGES = [{"role": "system", "content": "Роль"}, {"role": "user", "content": "Вопрос"}]


class CloudProviderTests(unittest.TestCase):
    def test_yandex_sends_matching_folder_and_project_headers(self):
        with api_server([(200, CHAT)]) as (url, requests):
            with patch.object(CloudProvider, "is_yandex", new_callable=PropertyMock, return_value=True):
                provider = CloudProvider(url, "test-key", "model", project="folder-test")
                provider.complete(MESSAGES)
            headers = {key.lower(): value for key, value in requests[0]["headers"].items()}
            self.assertEqual(headers["authorization"], "Api-Key test-key")
            self.assertEqual(headers["x-folder-id"], "folder-test")
            self.assertEqual(headers["openai-project"], "folder-test")

    def test_chat_endpoint_auth_and_usage(self):
        with api_server([(200, CHAT)]) as (url, requests):
            provider = CloudProvider(url, "secret-for-test", "test-model")
            self.assertEqual(provider.complete(MESSAGES, temperature=0.5, max_tokens=123), "Ответ")
            self.assertEqual(requests[0]["path"], "/v1/chat/completions")
            self.assertEqual(requests[0]["body"]["messages"], MESSAGES)
            self.assertEqual(requests[0]["body"]["max_tokens"], 123)
            self.assertEqual(requests[0]["headers"]["Authorization"], "Bearer secret-for-test")
            self.assertEqual(provider.calls[0]["usage"]["completion_tokens"], 3)
            self.assertNotIn("secret-for-test", repr(provider))

    def test_responses_endpoint_extracts_output_text(self):
        body = {
            "status": "completed",
            "output": [
                {"type": "reasoning", "summary": []},
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "Первая"},
                        {"type": "output_text", "text": "Вторая"},
                    ],
                },
            ],
        }
        with api_server([(200, body)]) as (url, requests):
            provider = CloudProvider(url, "test-key", "test-model", api_kind="responses")
            self.assertEqual(provider.complete(MESSAGES), "Первая\nВторая")
            self.assertEqual(requests[0]["path"], "/v1/responses")
            self.assertEqual(requests[0]["body"]["input"], MESSAGES)
            self.assertIn("max_output_tokens", requests[0]["body"])
            self.assertFalse(requests[0]["body"]["store"])

    def test_transient_retry_and_attempts_are_bounded(self):
        with api_server([(429, {}), (503, {}), (200, CHAT)]) as (url, requests):
            with patch("theatre.providers.time.sleep") as sleep:
                provider = CloudProvider(url, "test-key", "model")
                self.assertEqual(provider.complete(MESSAGES), "Ответ")
            self.assertEqual(len(requests), 3)
            self.assertEqual(provider.calls[0]["attempts"], 3)
            self.assertEqual(sleep.call_count, 2)

    def test_nonretry_auth_error_never_exposes_remote_body(self):
        with api_server([(401, {"error": "secret-key confidential prompt"})]) as (url, requests):
            provider = CloudProvider(url, "secret-key", "model")
            with self.assertRaises(ProviderError) as raised:
                provider.complete(MESSAGES)
            self.assertIn("401", str(raised.exception))
            self.assertNotIn("secret-key", str(raised.exception))
            self.assertNotIn("confidential", str(raised.exception))
            self.assertEqual(len(requests), 1)
            self.assertEqual(provider.calls, [])

    def test_exhausted_rate_limit_does_not_retry_forever(self):
        with api_server([(429, {}), (429, {})]) as (url, requests):
            provider = CloudProvider(url, "key", "model", max_attempts=2)
            with patch("theatre.providers.time.sleep"), self.assertRaisesRegex(ProviderError, "attempts=2"):
                provider.complete(MESSAGES)
            self.assertEqual(len(requests), 2)

    def test_incomplete_output_is_explicit_failure(self):
        cases = [
            ("chat", {"choices": [{"finish_reason": "length", "message": {"content": "partial"}}]}),
            ("responses", {"status": "incomplete", "output": []}),
            ("chat", {"choices": []}),
        ]
        for kind, body in cases:
            with (
                self.subTest(kind=kind, body=body),
                api_server([(200, body)]) as (url, _),
                self.assertRaises(ProviderError),
            ):
                CloudProvider(url, "key", "model", api_kind=kind).complete(MESSAGES)

    def test_yandex_legacy_env_and_model_expansion(self):
        provider = CloudProvider.from_env({"folder_id": "folder-test", "api_key": "not-a-real-key"})
        self.assertEqual(provider.model, "gpt://folder-test/aliceai-llm-flash")
        self.assertEqual(provider.api_kind, "responses")
        self.assertEqual(provider.auth_scheme, "Api-Key")
        provider = CloudProvider.from_env(
            {"YANDEX_FOLDER_ID": "f", "LLM_API_KEY": "k", "LLM_MODEL": "yandexgpt"}
        )
        self.assertEqual(provider.model, "gpt://f/yandexgpt")

    def test_no_env_mutation_and_no_silent_provider_substitution(self):
        env = {"LLM_BASE_URL": "https://example.org/v1", "LLM_API_KEY": "key", "LLM_MODEL": "model"}
        original = dict(env)
        provider = CloudProvider.from_env(env)
        self.assertEqual(env, original)
        self.assertEqual(provider.api_kind, "chat")
        with self.assertRaises(ProviderError):
            CloudProvider.from_env({})

    def test_key_and_url_safety_and_request_limits(self):
        for url in [
            "http://remote.example/v1",
            "https://user:pass@example.org/v1",
            "https://example.org/v1?key=foo",
        ]:
            with self.subTest(url=url), self.assertRaises(ValueError):
                CloudProvider(url, "key", "model")
        provider = CloudProvider("http://localhost:1234/v1", "key", "model")
        for messages, temperature, tokens in [
            ([], 0.7, 10),
            (MESSAGES, float("nan"), 10),
            (MESSAGES, 0.7, 0),
            (MESSAGES, 0.7, True),
        ]:
            with self.assertRaises(ValueError):
                provider.complete(messages, temperature=temperature, max_tokens=tokens)
        self.assertEqual(_retry_delay("999999", 0), 15)
        self.assertEqual(_retry_delay("-5", 0), 0)


class FakeInputs(dict):
    def __init__(self, tokens):
        super().__init__(input_ids=SimpleNamespace(shape=(1, tokens)))

    def to(self, device):
        self.target_device = device
        return self


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(self, messages, **kwargs):
        self.template_kwargs = kwargs
        return "rendered chat"

    def __call__(self, text, return_tensors):
        return FakeInputs(3)

    def decode(self, tokens, skip_special_tokens):
        self.decoded = tokens
        return "Ответ локальной модели"


class FakeModel:
    device = "cpu"
    config = SimpleNamespace(max_position_embeddings=32768)

    def generate(self, **kwargs):
        self.generation_kwargs = kwargs
        return [[100, 101, 102, 201, 202]]


class LocalProviderTests(unittest.TestCase):
    def make_provider(self):
        provider = HFLocalProvider()
        provider._tokenizer = FakeTokenizer()
        provider._model = FakeModel()
        provider._torch = SimpleNamespace(inference_mode=nullcontext)
        return provider

    def test_local_is_lazy_and_explicitly_identified(self):
        provider = HFLocalProvider()
        self.assertFalse(provider.loaded)
        self.assertEqual(provider.name, "local-hf:Qwen/Qwen3-4B-Instruct-2507")

    def test_quantization_selection_is_explicit_and_validated(self):
        self.assertEqual(HFLocalProvider().quantization, "none")
        self.assertEqual(HFLocalProvider(quantization="nf4").quantization, "nf4")
        for invalid in ("int8", "", None, ["nf4"]):
            with self.subTest(value=invalid), self.assertRaisesRegex(ValueError, "quantization"):
                HFLocalProvider(quantization=invalid)

    def test_nf4_cannot_silently_fall_back_to_cpu(self):
        tokenizer_loader = Mock()
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        fake_transformers = SimpleNamespace(AutoTokenizer=tokenizer_loader, AutoModelForCausalLM=Mock())
        with patch.dict("sys.modules", {"torch": fake_torch, "transformers": fake_transformers}):
            with self.assertRaisesRegex(ProviderError, "NF4 quantization requires CUDA"):
                HFLocalProvider(quantization="nf4", device="cpu")._load()
        tokenizer_loader.from_pretrained.assert_not_called()

    def test_nf4_load_uses_requested_quantizer_and_bfloat16_compute(self):
        quantizer_factory = Mock(return_value=object())
        model = SimpleNamespace(config=SimpleNamespace(_commit_hash="test-snapshot"), eval=Mock())
        model_loader = SimpleNamespace(from_pretrained=Mock(return_value=model))
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True),
            bfloat16="test-bfloat16",
            float32="test-float32",
            manual_seed=Mock(),
        )
        fake_transformers = SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=Mock()),
            AutoModelForCausalLM=model_loader,
            BitsAndBytesConfig=quantizer_factory,
        )
        with patch.dict("sys.modules", {"torch": fake_torch, "transformers": fake_transformers}):
            provider = HFLocalProvider(quantization="nf4", local_files_only=True)
            provider._load()
        quantizer_factory.assert_called_once_with(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype="test-bfloat16",
        )
        self.assertIs(
            model_loader.from_pretrained.call_args.kwargs["quantization_config"],
            quantizer_factory.return_value,
        )
        self.assertEqual(provider.resolved_revision, "test-snapshot")

    def test_only_new_tokens_decoded_and_thinking_disabled(self):
        provider = self.make_provider()
        answer = provider.complete(MESSAGES, temperature=0.7, max_tokens=123)
        self.assertEqual(answer, "Ответ локальной модели")
        self.assertEqual(provider._tokenizer.decoded, [201, 202])
        self.assertFalse(provider._tokenizer.template_kwargs["enable_thinking"])
        self.assertEqual(provider._model.generation_kwargs["max_new_tokens"], 123)
        self.assertEqual(provider.calls[0]["usage"], {"prompt_tokens": 3, "completion_tokens": 2})
        self.assertEqual(provider.calls[0]["quantization"], "none")

    def test_cuda_cache_released_after_successful_request(self):
        provider = self.make_provider()
        provider._model.device = "cuda:0"
        provider._torch.cuda = SimpleNamespace(empty_cache=Mock())
        provider.complete(MESSAGES)
        provider._torch.cuda.empty_cache.assert_called_once_with()

    def test_cuda_cache_released_after_model_error(self):
        provider = self.make_provider()
        provider._model.device = "cuda:0"
        provider._model.generate = Mock(side_effect=RuntimeError("CUDA allocation failed"))
        provider._torch.cuda = SimpleNamespace(empty_cache=Mock())
        with self.assertRaisesRegex(ProviderError, "Local inference failed"):
            provider.complete(MESSAGES)
        provider._torch.cuda.empty_cache.assert_called_once_with()
        self.assertEqual(provider.calls, [])

    def test_cpu_request_does_not_touch_cuda_cache(self):
        provider = self.make_provider()
        provider._torch.cuda = SimpleNamespace(empty_cache=Mock())
        provider.complete(MESSAGES)
        provider._torch.cuda.empty_cache.assert_not_called()

    def test_greedy_mode_omits_temperature_and_top_p(self):
        provider = self.make_provider()
        provider.complete(MESSAGES, temperature=0)
        self.assertFalse(provider._model.generation_kwargs["do_sample"])
        self.assertNotIn("temperature", provider._model.generation_kwargs)
        self.assertNotIn("top_p", provider._model.generation_kwargs)

    def test_context_overflow_rejected_instead_of_truncating_prompt(self):
        provider = self.make_provider()
        provider._model.config = SimpleNamespace(max_position_embeddings=10)
        with self.assertRaisesRegex(ProviderError, "context budget"):
            provider.complete(MESSAGES, max_tokens=8)
        self.assertEqual(provider.calls, [])


if __name__ == "__main__":
    unittest.main()
