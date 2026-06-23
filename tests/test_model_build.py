import pytest

from trust_the_eval.model import build_model
from trust_the_eval.model.providers import AnthropicClient, LocalServerClient, OpenAIClient


def test_build_model_api_spec_builds_openai_compatible_client():
    c = build_model("api:http://localhost:11434/v1#llama3.2", api_key="sk-x")
    assert isinstance(c, LocalServerClient)
    assert c.base_url == "http://localhost:11434/v1"
    assert c.model == "llama3.2"
    assert c.api_key == "sk-x"


def test_build_model_server_spec_still_supported():
    c = build_model("server:http://localhost:1234/v1#mistral")
    assert isinstance(c, LocalServerClient)
    assert c.base_url == "http://localhost:1234/v1"
    assert c.model == "mistral"


def test_build_model_legacy_provider_specs_still_supported():
    assert isinstance(build_model("anthropic:claude-sonnet-4-20250514"), AnthropicClient)
    assert isinstance(build_model("openai:gpt-4o-mini"), OpenAIClient)


def test_build_model_api_spec_validates_shape():
    with pytest.raises(ValueError, match="api spec must be"):
        build_model("api:http://localhost:11434/v1")
    with pytest.raises(ValueError, match="api spec needs both"):
        build_model("api:#llama3.2")
