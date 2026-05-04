"""
OpenRouter client and provider router.

Why OpenRouter: one API key, swap models without code changes, pay
per-token across providers. Critical for the cost structure of a
self-improving agent — the background skill-review fork runs after
every turn, and routing it to a cheap model is the difference between
$50/month and $500/month in token costs.

Three roles:
    "smart"      -- main agent reasoning (default: Claude Sonnet 4.6)
    "cheap"      -- background review, summarization (default: Haiku 4.5)
    "embeddings" -- semantic memory search (default: a cheap embedding model)

Override via env or config:
    ATHENA_MODEL_SMART     = "anthropic/claude-sonnet-4.6"
    ATHENA_MODEL_CHEAP     = "anthropic/claude-haiku-4.5"
    ATHENA_MODEL_EMBED     = "openai/text-embedding-3-small"
    OPENROUTER_API_KEY     = "sk-or-..."
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal, Optional


Role = Literal["smart", "cheap", "embeddings"]

DEFAULT_MODELS: dict[Role, str] = {
    "smart": "anthropic/claude-sonnet-4.6",
    "cheap": "anthropic/claude-haiku-4.5",
    "embeddings": "openai/text-embedding-3-small",
}


@dataclass
class LLMConfig:
    api_key: str = ""
    base_url: str = "https://openrouter.ai/api/v1"
    models: dict[Role, str] = field(default_factory=dict)
    referer: str = "https://localhost/athena"   # OpenRouter ranks by referer
    app_name: str = "athena-agent"
    timeout: float = 60.0
    max_retries: int = 3

    @classmethod
    def from_env(cls) -> "LLMConfig":
        models = dict(DEFAULT_MODELS)
        for role in ("smart", "cheap", "embeddings"):
            env_key = f"ATHENA_MODEL_{role.upper()}"
            if env_key in os.environ:
                models[role] = os.environ[env_key]  # type: ignore
        return cls(
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            base_url=os.environ.get("OPENROUTER_BASE_URL",
                                    "https://openrouter.ai/api/v1"),
            models=models,
            referer=os.environ.get("ATHENA_REFERER",
                                   "https://localhost/athena"),
            app_name=os.environ.get("ATHENA_APP_NAME", "athena-agent"),
        )


@dataclass
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[list[dict]] = None

    def to_openai(self) -> dict:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        return d


@dataclass
class Completion:
    text: str
    tool_calls: list[dict] = field(default_factory=list)
    model: str = ""
    usage: dict = field(default_factory=dict)
    finish_reason: str = ""
    raw: dict = field(default_factory=dict)


class LLM:
    """Thin OpenRouter client. Synchronous; streaming optional."""

    def __init__(self, config: Optional[LLMConfig] = None):
        try:
            import requests  # noqa: F401
        except ImportError as e:
            raise ImportError("requests not installed. pip install requests") from e
        import requests
        self._requests = requests
        self.config = config or LLMConfig.from_env()
        if not self.config.api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. "
                "Get a key at https://openrouter.ai/keys"
            )
        # Final models map (defaults filled in)
        self._models = {**DEFAULT_MODELS, **self.config.models}

    def model_for(self, role: Role) -> str:
        return self._models[role]

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "HTTP-Referer": self.config.referer,
            "X-Title": self.config.app_name,
            "Content-Type": "application/json",
        }

    # ---- Chat completions ----------------------------------------------

    def chat(
        self,
        messages: list[Message],
        *,
        role: Role = "smart",
        model: Optional[str] = None,
        tools: Optional[list[dict]] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        response_format: Optional[dict] = None,
    ) -> Completion:
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        payload: dict[str, Any] = {
            "model": model or self.model_for(role),
            "messages": [m.to_openai() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if response_format:
            payload["response_format"] = response_format

        last_err: Optional[Exception] = None
        for attempt in range(self.config.max_retries):
            try:
                r = self._requests.post(
                    url, headers=self._headers(),
                    json=payload, timeout=self.config.timeout,
                )
            except self._requests.RequestException as e:
                last_err = e
                time.sleep(0.5 * (2 ** attempt))
                continue
            if r.status_code == 429:
                # backoff
                time.sleep(1.0 * (2 ** attempt))
                last_err = RuntimeError("openrouter 429")
                continue
            if r.status_code >= 500:
                time.sleep(0.5 * (2 ** attempt))
                last_err = RuntimeError(f"openrouter {r.status_code}")
                continue
            if not r.ok:
                raise RuntimeError(
                    f"openrouter {r.status_code}: {r.text[:300]}"
                )
            data = r.json()
            choice = (data.get("choices") or [{}])[0]
            msg = choice.get("message", {}) or {}
            return Completion(
                text=msg.get("content") or "",
                tool_calls=msg.get("tool_calls") or [],
                model=data.get("model", payload["model"]),
                usage=data.get("usage", {}) or {},
                finish_reason=choice.get("finish_reason", ""),
                raw=data,
            )
        raise RuntimeError(f"openrouter failed after retries: {last_err}")

    # ---- Embeddings ----------------------------------------------------

    def embed(self, texts: list[str], *,
              model: Optional[str] = None) -> list[list[float]]:
        url = f"{self.config.base_url.rstrip('/')}/embeddings"
        payload = {
            "model": model or self.model_for("embeddings"),
            "input": texts,
        }
        r = self._requests.post(url, headers=self._headers(),
                                json=payload, timeout=self.config.timeout)
        if not r.ok:
            raise RuntimeError(f"openrouter embed {r.status_code}: {r.text[:300]}")
        data = r.json()
        return [item["embedding"] for item in data.get("data", [])]
