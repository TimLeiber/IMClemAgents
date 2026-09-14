"""Native claude code configuration and execution helpers."""

from typing import Any

from ..utils import model_connection as connections


def _anthropic_proxy_base_url(proxy_base_url: str) -> str:
    """Let Anthropic clients append /v1/messages exactly once."""
    return proxy_base_url[:-3] if proxy_base_url.endswith("/v1") else proxy_base_url


def _resolve_model_connection(model_spec: dict[str, Any]) -> dict[str, Any]:
    clem_model = model_spec["model_name"]
    backend = model_spec.get("backend")

    if backend == "openai_compatible":
        key_config = connections._openai_compatible_key_config()
        connection = connections._openai_compatible_common(model_spec, key_config)
        connection.update({"harness": "claude_code",
                           "env": {"ANTHROPIC_BASE_URL": connection["base_url"],
                                   "ANTHROPIC_AUTH_TOKEN": connections._openai_compatible_api_key(key_config),
                                   "ANTHROPIC_API_KEY": "",
                                   "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}})
        return connection

    if backend != "openrouter":
        raise NotImplementedError("MVP limitation: Claude Code registry-model support currently only "
                                  f"supports clembench models with backend='openrouter' or backend='openai_compatible'. "
                                  f"Model {clem_model!r} has backend={backend!r}.")

    model_id = model_spec.get("model_id") or model_spec.get("model_name")
    key_config = connections._openrouter_key_config()

    return {"harness": "claude_code",
            "clem_model": model_spec["model_name"],
            "backend": "openrouter",
            "model": model_id,
            "display_model": model_spec["model_name"],
            "base_url": _openrouter_anthropic_base_url(key_config),
            "env": {"ANTHROPIC_BASE_URL": _openrouter_anthropic_base_url(key_config),
                    "ANTHROPIC_AUTH_TOKEN": connections._openrouter_api_key(key_config),
                    "ANTHROPIC_API_KEY": "",
                    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}}


def _openrouter_anthropic_base_url(config: dict[str, Any]) -> str:
    base_url = str(config.get("base_url") or "https://openrouter.ai/api").rstrip("/")

    # clemcore/openrouter configs often use the openai-compatible /api/v1
    # endpoint. claude code needs openrouter's anthropic-compatible endpoint
    if base_url.endswith("/api/v1"):
        return base_url[:-3]

    return base_url
