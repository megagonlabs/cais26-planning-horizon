from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from dotenv import load_dotenv
import os

from .openai import OpenAIClient
from .vertexai_openai import VertexAIOpenAIClient
from .vllm import vLLMClient


@dataclass
class LLMProviderSpec:
    provider_id: str
    type: str  # "openai" | "vllm" | ...
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None
    base_url: Optional[str] = None
    organization: Optional[str] = None
    timeout: Optional[int] = None
    max_retries: Optional[int] = None
    project_id_env: Optional[str] = None  # For Vertex AI
    location_env: Optional[str] = None  # For Vertex AI


class LLMProviderRegistry:
    def __init__(self) -> None:
        self._provider_specs: dict[str, LLMProviderSpec] = {}
        self._clients: dict[str, Any] = {}

    @classmethod
    def from_config(cls, providers_cfg: dict[str, Any]) -> "LLMProviderRegistry":
        load_dotenv(override=True)
        reg = cls()
        for pid, spec in providers_cfg.items():
            spec = LLMProviderSpec(
                provider_id=pid,
                type=spec.get("type", "openai"),
                api_key=spec.get("api_key"),
                api_key_env=spec.get("api_key_env"),
                base_url=spec.get("base_url"),
                organization=spec.get("organization"),
                timeout=spec.get("timeout"),
                max_retries=spec.get("max_retries"),
                project_id_env=spec.get("project_id_env"),
                location_env=spec.get("location_env"),
            )
            reg._register_provider_spec(spec)
        return reg

    def get_call(self, provider_id: str) -> Callable:
        if provider_id not in self._provider_specs:
            raise KeyError(f"Unknown LLM provider: {provider_id}")
        if provider_id not in self._clients:
            self._clients[provider_id] = self._build_client(self._provider_specs[provider_id])
        return self._clients[provider_id].call

    def _register_provider_spec(self, spec: LLMProviderSpec) -> None:
        self._provider_specs[spec.provider_id] = spec

    def _build_client(self, spec: LLMProviderSpec) -> Any:
        api_key = os.getenv(spec.api_key_env) if spec.api_key_env else spec.api_key
        kwargs: dict[str, Any] = {}

        if spec.type == "openai":
            if api_key:
                kwargs["api_key"] = api_key
            if spec.base_url:
                kwargs["base_url"] = spec.base_url
            if spec.organization:
                kwargs["organization"] = spec.organization
            if spec.timeout:
                kwargs["timeout"] = spec.timeout
            if spec.max_retries:
                kwargs["max_retries"] = spec.max_retries
            client = OpenAIClient(**kwargs)
        elif spec.type == "fireworks":
            if api_key:
                kwargs["api_key"] = api_key
            if spec.base_url:
                kwargs["base_url"] = spec.base_url
            client = OpenAIClient(**kwargs)
        elif spec.type == "vertexai-openai":  # OpenAI-compatible Vertex AI endpoint
            project_id_env = spec.project_id_env or "GOOGLE_CLOUD_PROJECT"
            location_env = spec.location_env or "GOOGLE_CLOUD_LOCATION"
            project_id = os.getenv(project_id_env, "")
            location = os.getenv(location_env, "global")
            client = VertexAIOpenAIClient(
                project_id=project_id,
                location=location,
                timeout=spec.timeout,
                max_retries=spec.max_retries,
            )
        elif spec.type == "vllm":
            if not spec.base_url:
                raise ValueError(f"vLLM provider {spec.provider_id} requires base_url")
            kwargs["base_url"] = spec.base_url
            if api_key:
                kwargs["api_key"] = api_key
            client = vLLMClient(**kwargs)
        else:
            raise ValueError(f"Unknown provider type: {spec.type}")

        return client

    def close_all(self) -> None:
        """Close all provider connections."""
        for provider in self._clients.values():
            try:
                if hasattr(provider, "close"):
                    provider.close()
            except Exception:
                pass

    def __del__(self):
        try:
            self.close_all()
        except Exception:
            pass
