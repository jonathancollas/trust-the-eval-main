"""Model clients: real providers and controllable local (demo) models.

`build_model` is the single place that turns a CLI/UI spec string into a
ModelClient, so the CLI and the web UI share one code path.
"""
from __future__ import annotations
from typing import Optional

from ..probe import ModelClient


def build_model(spec: Optional[str], api_key: Optional[str] = None) -> Optional[ModelClient]:
    """Spec -> ModelClient. None/'' /'none' -> None (static-only battery).

      local:<name>              a controllable DEMO model (honest, sandbagger, ...)
                                synthetic; for trying the tool, not a real model
      api:<base_url>#<model>    any OpenAI-compatible endpoint (OpenAI, Ollama,
                                LM Studio, vLLM, llama.cpp, or a remote gateway)
      server:<base_url>#<model> legacy alias of `api:...`
      anthropic:<model>         backward-compatible legacy path (official SDK)
      openai:<model>            backward-compatible legacy path (official SDK)

    `api_key`, when given (e.g. typed into the UI), is used for this call only and
    takes precedence over the environment. It is never persisted by this function.
    Real providers fall back to ANTHROPIC_API_KEY / OPENAI_API_KEY if no key here.
    """
    if not spec or spec == "none":
        return None
    if spec.startswith("local:"):
        from .local import from_spec
        return from_spec(spec)
    if spec.startswith("anthropic:"):
        from .providers import AnthropicClient
        return AnthropicClient(model=spec.split(":", 1)[1] or "claude-sonnet-4-20250514",
                               api_key=api_key)
    if spec.startswith("openai:"):
        from .providers import OpenAIClient
        return OpenAIClient(model=spec.split(":", 1)[1] or "gpt-4o-mini",
                            api_key=api_key)
    if spec.startswith("api:") or spec.startswith("server:"):
        from .providers import LocalServerClient
        rest = spec.split(":", 1)[1]
        if "#" not in rest:
            raise ValueError("api spec must be 'api:<base_url>#<model>' "
                             "(legacy: server:<base_url>#<model>)")
        base_url, model = rest.rsplit("#", 1)
        if not base_url or not model:
            raise ValueError("api spec needs both a base URL and a model name")
        return LocalServerClient(base_url=base_url, model=model, api_key=api_key)
    raise ValueError(f"unknown model spec '{spec}' (use local:<name>, "
                     "api:<base_url>#<model>, anthropic:<model>, "
                     "openai:<model>, server:<base_url>#<model>)")
