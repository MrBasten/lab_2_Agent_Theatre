"""Real cloud and local LLM adapters; imports do not download models or use keys.

Sources: https://aistudio.yandex.ru/en/docs/ai-studio/operations/generation/create-prompt
and https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 (non-thinking instruction model).
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


class ProviderError(RuntimeError):
    """A real model failed. Callers must not silently substitute fixture text."""


def _validate_request(messages: list[dict], temperature: float, max_tokens: int) -> None:
    if not messages or any(
        not isinstance(m, dict)
        or m.get("role") not in {"system", "developer", "user", "assistant"}
        or not isinstance(m.get("content"), str)
        for m in messages
    ):
        raise ValueError("messages must contain role/content text dictionaries")
    if not math.isfinite(temperature) or not 0 <= temperature <= 2:
        raise ValueError("temperature must be finite and between 0 and 2")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")


def _retry_delay(value: str | None, attempt: int) -> float:
    """Respect Retry-After seconds or HTTP dates, bounded to keep retries finite."""
    fallback = float(2**attempt)
    if value:
        try:
            delay = float(value)
        except ValueError:
            try:
                target = parsedate_to_datetime(value)
                if target.tzinfo is None:
                    target = target.replace(tzinfo=UTC)
                delay = (target - datetime.now(UTC)).total_seconds()
            except (ValueError, TypeError, OverflowError):
                delay = fallback
        if math.isfinite(delay):
            return min(15.0, max(0.0, delay))
    return min(15.0, fallback)


@dataclass
class CloudProvider:
    """A bounded, non-streaming OpenAI-compatible HTTP client.

    ``api_kind`` explicitly chooses /chat/completions or /responses. Retries apply
    only to transport failures, 429 and 5xx; an API may bill a lost response twice.
    Error messages deliberately omit server bodies, request URLs and credentials.
    """

    base_url: str
    api_key: str = field(repr=False)
    model: str
    api_kind: str = "chat"
    project: str | None = None
    timeout: float = 90.0
    max_attempts: int = 3
    auth_scheme: str | None = None
    calls: list[dict] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        parts = urlsplit(self.base_url)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            raise ValueError("LLM_BASE_URL must be an HTTP(S) API base URL")
        if parts.username or parts.password or parts.query or parts.fragment:
            raise ValueError("LLM_BASE_URL must not embed credentials, query or fragment")
        if parts.scheme == "http" and parts.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Use HTTPS for cloud APIs; HTTP is allowed only on loopback")
        if self.api_kind not in {"chat", "responses"}:
            raise ValueError("LLM_API_KIND must be chat or responses")
        if not self.api_key or self.api_key.strip() in {"...", "YOUR_API_KEY", "<API_key>"}:
            raise ProviderError("Set LLM_API_KEY (or api_key) in the local .env file")
        if not self.model or self.model == "...":
            raise ProviderError("Set LLM_MODEL to a model available in your account")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be a positive finite number")
        if not 1 <= self.max_attempts <= 5:
            raise ValueError("max_attempts must be between 1 and 5")
        if self.auth_scheme is None:
            self.auth_scheme = "Api-Key" if self.is_yandex else "Bearer"
        if self.auth_scheme not in {"Api-Key", "Bearer"}:
            raise ValueError("auth_scheme must be Api-Key or Bearer")

    @property
    def is_yandex(self) -> bool:
        return urlsplit(self.base_url).hostname == "ai.api.cloud.yandex.net"

    @property
    def name(self) -> str:
        model = self.model
        if self.is_yandex and model.startswith("gpt://"):
            _, separator, slug = model[6:].partition("/")
            if separator:
                model = f"gpt://<folder_id>/{slug}"
        return f"cloud:{model}"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, *, model: str | None = None) -> CloudProvider:
        env = os.environ if env is None else env
        base_url = env.get("LLM_BASE_URL", "https://ai.api.cloud.yandex.net/v1")
        yandex = urlsplit(base_url).hostname == "ai.api.cloud.yandex.net"
        project = env.get("YANDEX_FOLDER_ID") or env.get("folder_id") or None
        selected = model or env.get("LLM_MODEL", "")
        if not selected and yandex and project:
            selected = f"gpt://{project}/aliceai-llm-flash"
        elif selected and yandex and project and not selected.startswith("gpt://"):
            selected = f"gpt://{project}/{selected}"
        return cls(
            base_url=base_url,
            api_key=env.get("LLM_API_KEY") or env.get("YANDEX_API_KEY") or env.get("api_key", ""),
            model=selected,
            api_kind=env.get("LLM_API_KIND", "responses" if yandex else "chat"),
            project=project,
            timeout=float(env.get("LLM_TIMEOUT", "90")),
            max_attempts=int(env.get("LLM_MAX_ATTEMPTS", "3")),
        )

    def complete(self, messages: list[dict], *, temperature: float = 0.7, max_tokens: int = 1000) -> str:
        _validate_request(messages, temperature, max_tokens)
        if self.api_kind == "responses":
            payload = {
                "model": self.model,
                "input": messages,
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "store": False,
            }
            endpoint = "/responses"
        else:
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            endpoint = "/chat/completions"
        headers = {
            "Authorization": f"{self.auth_scheme} {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.project:
            headers["OpenAI-Project"] = self.project
            if self.is_yandex:
                # The auth reference documents OpenAI-Project; the JavaScript
                # quickstart additionally sends x-folder-id. Both identify the
                # same folder, avoiding conflicting routing between examples.
                headers["x-folder-id"] = self.project
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        started = time.monotonic()
        for attempt in range(self.max_attempts):
            request = Request(self.base_url + endpoint, data=data, headers=headers, method="POST")
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    # Limit response size; generated short scenes cannot legitimately need >8 MiB.
                    raw = response.read(8 * 1024 * 1024 + 1)
                if len(raw) > 8 * 1024 * 1024:
                    raise ProviderError("LLM response exceeded the 8 MiB limit")
                body = json.loads(raw.decode("utf-8"))
                text = self._extract_text(body)
                self.calls.append(
                    {
                        "provider": self.name,
                        "seconds": round(time.monotonic() - started, 3),
                        "attempts": attempt + 1,
                        "usage": body.get("usage", {}),
                    }
                )
                return text
            except HTTPError as error:
                status = error.code
                retry_after = error.headers.get("Retry-After") if error.headers else None
                error.close()
                if (status == 429 or 500 <= status <= 599) and attempt + 1 < self.max_attempts:
                    time.sleep(_retry_delay(retry_after, attempt))
                    continue
                hint = {
                    401: "check the API key",
                    403: "check API permissions and project",
                    404: "check base URL, API kind and model",
                    429: "quota or rate limit reached",
                }.get(status, "check provider availability and model settings")
                raise ProviderError(f"LLM HTTP {status}: {hint}; attempts={attempt + 1}") from None
            except (URLError, TimeoutError, ConnectionError):
                if attempt + 1 < self.max_attempts:
                    time.sleep(_retry_delay(None, attempt))
                    continue
                raise ProviderError(
                    f"LLM connection failed after {attempt + 1} attempts; check network/timeout"
                ) from None
            except (ValueError, KeyError, TypeError, IndexError):
                raise ProviderError("LLM returned an invalid or incomplete JSON response") from None
        raise ProviderError("LLM retry budget exhausted")  # Defensive; loop always returns or raises.

    def _extract_text(self, body: dict) -> str:
        if self.api_kind == "responses":
            if body.get("status") in {"failed", "incomplete", "cancelled"}:
                raise ProviderError(
                    "LLM response did not complete; increase max_tokens or inspect provider limits"
                )
            chunks = [
                part["text"]
                for item in body.get("output", [])
                if item.get("type") == "message"
                for part in item.get("content", [])
                if part.get("type") == "output_text"
            ]
            text = "\n".join(chunks)
        else:
            choice = body["choices"][0]
            if choice.get("finish_reason") == "length":
                raise ProviderError("LLM output was truncated; increase max_tokens")
            text = choice["message"]["content"]
        if not isinstance(text, str) or not text.strip():
            raise ProviderError("LLM returned no text (refusal or unsupported response format)")
        return text.strip()


def create_cloud_provider(env: Mapping[str, str] | None = None, *, model: str | None = None) -> CloudProvider:
    return CloudProvider.from_env(env, model=model)


class HFLocalProvider:
    """Runs open weights from Hugging Face locally with no remote-code execution.

    Models are lazy-loaded on the first call. CUDA is preferred when available;
    CPU remains supported but is substantially slower. The first load downloads
    model weights (Qwen3-4B-Instruct-2507 is about 8.1 GB). No cloud inference fallback exists.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-4B-Instruct-2507",
        *,
        device: str = "auto",
        seed: int = 42,
        max_input_tokens: int = 8192,
        revision: str | None = None,
        local_files_only: bool = False,
        quantization: str = "none",
    ) -> None:
        if device not in {"auto", "cuda", "cpu"}:
            raise ValueError("device must be auto, cuda or cpu")
        if max_input_tokens < 128:
            raise ValueError("max_input_tokens must be at least 128")
        if quantization not in ("none", "nf4"):
            raise ValueError("quantization must be none or nf4")
        self.model_id, self.device, self.seed = model_id, device, seed
        self.max_input_tokens, self.revision = max_input_tokens, revision
        self.local_files_only = local_files_only
        self.quantization = quantization
        self.resolved_revision: str | None = revision
        self.calls: list[dict] = []
        self._model = self._tokenizer = self._torch = None

    @property
    def name(self) -> str:
        return f"local-hf:{self.model_id}"

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _load(self) -> None:
        if self.loaded:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ProviderError(
                "Local inference needs torch, transformers>=4.51, accelerate and safetensors; install the local extra"
            ) from None
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device != "auto":
            device = self.device
        if device == "cuda" and not torch.cuda.is_available():
            raise ProviderError("CUDA requested but unavailable; install a compatible CUDA PyTorch build")
        if self.quantization == "nf4" and device != "cuda":
            raise ProviderError("NF4 quantization requires CUDA; choose quantization='none' for CPU")
        options = {"trust_remote_code": False, "local_files_only": self.local_files_only}
        if self.revision:
            options["revision"] = self.revision
        try:
            model_options = {}
            if self.quantization == "nf4":
                from transformers import BitsAndBytesConfig

                model_options["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            tokenizer = AutoTokenizer.from_pretrained(self.model_id, **options)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map="auto" if device == "cuda" else "cpu",
                torch_dtype="auto" if device == "cuda" else torch.float32,
                **model_options,
                **options,
            )
            model.eval()
            torch.manual_seed(self.seed)
        except (OSError, RuntimeError, ValueError, TypeError, ImportError, KeyError, AttributeError) as error:
            # Exception class helps diagnose OOM/dependency failures without leaking hub tokens.
            raise ProviderError(
                f"Local model load failed ({type(error).__name__}); check weights, disk, memory and dependencies (NF4 requires compatible bitsandbytes)"
            ) from None
        self._tokenizer, self._model, self._torch = tokenizer, model, torch
        # Record the snapshot actually used, even with newer Transformers versions
        # whose config no longer retains _commit_hash.
        self.resolved_revision = getattr(model.config, "_commit_hash", None)
        if not self.resolved_revision:
            try:
                from transformers.utils.hub import cached_file

                config_path = Path(
                    cached_file(self.model_id, "config.json", revision=self.revision, local_files_only=True)
                )
                parts = config_path.parts
                if "snapshots" in parts:
                    self.resolved_revision = parts[parts.index("snapshots") + 1]
            except (OSError, ValueError, TypeError, ImportError):
                pass  # Provenance may be unavailable for a user-supplied local directory.
        self.resolved_revision = self.resolved_revision or self.revision

    def complete(self, messages: list[dict], *, temperature: float = 0.7, max_tokens: int = 1000) -> str:
        _validate_request(messages, temperature, max_tokens)
        self._load()
        started = time.monotonic()
        inputs = output = new_tokens = None
        using_cuda = str(self._model.device).startswith("cuda")
        try:
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)
            prompt_tokens = inputs["input_ids"].shape[-1]
            model_limit = getattr(self._model.config, "max_position_embeddings", 32768)
            if prompt_tokens > self.max_input_tokens or prompt_tokens + max_tokens > model_limit:
                raise ProviderError(
                    "Local prompt exceeds context budget; shorten history or lower max_tokens"
                )
            pad_id = self._tokenizer.pad_token_id
            generation = {
                "max_new_tokens": max_tokens,
                "do_sample": temperature > 0,
                "pad_token_id": pad_id if pad_id is not None else self._tokenizer.eos_token_id,
            }
            if temperature > 0:
                generation.update(temperature=temperature, top_p=0.8, top_k=20)
            with self._torch.inference_mode():
                output = self._model.generate(**inputs, **generation)
            new_tokens = output[0][prompt_tokens:]
            answer = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            if not answer:
                raise ProviderError("Local model returned no text")
            self.calls.append(
                {
                    "provider": self.name,
                    "seconds": round(time.monotonic() - started, 3),
                    "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": len(new_tokens)},
                    "device": str(self._model.device),
                    "revision": self.resolved_revision or self.revision or "unknown",
                    "quantization": self.quantization,
                }
            )
            return answer
        except ProviderError:
            raise
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError) as error:
            raise ProviderError(
                f"Local inference failed ({type(error).__name__}); check memory and model compatibility"
            ) from None
        finally:
            # These tensors may keep a full KV cache or prefill allocations alive.
            # Release them before clearing unused CUDA blocks between actor turns;
            # on Windows retained reserved blocks can trigger WDDM shared-RAM paging.
            del inputs, output, new_tokens
            if using_cuda:
                self._torch.cuda.empty_cache()


HuggingFaceProvider = HFLocalProvider
