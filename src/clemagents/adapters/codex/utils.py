"""Native codex configuration and execution helpers."""

from typing import Any

from ..utils import model_connection as connections


def _positive_token_limit(value: int | None, name: str) -> int | None:
    """Validate a native token count without silently converting invalid values."""
    if value is not None and (type(value) is not int or value <= 0):
        raise ValueError(f"{name} must be a positive integer token count")
    return value


def _resolve_model_connection(model_spec: dict[str, Any]) -> dict[str, Any]:
    clem_model = model_spec["model_name"]
    backend = model_spec.get("backend")

    model_id = model_spec.get("model_id") or model_spec.get("model_name")

    if backend == "openrouter":
        key_config = connections._openrouter_key_config()

        return {"harness": "codex",
                "clem_model": model_spec["model_name"],
                "backend": "openrouter",
                "model": model_id,
                "display_model": model_spec["model_name"],
                "base_url": connections._openrouter_openai_base_url(key_config),
                "env": {"OPENROUTER_API_KEY": connections._openrouter_api_key(key_config),
                        "OPENAI_API_KEY": connections._openrouter_api_key(key_config)}}

    if backend == "openai":
        key_config = connections._openai_key_config()

        return {"harness": "codex",
                "clem_model": model_spec["model_name"],
                "backend": "openai",
                "model": model_id,
                "display_model": model_spec["model_name"],
                "base_url": "https://api.openai.com/v1",
                "env": {"OPENAI_API_KEY": connections._openai_api_key(key_config)}}

    if backend == "openai_compatible":
        key_config = connections._openai_compatible_key_config()
        connection = connections._openai_compatible_common(model_spec, key_config)
        connection.update({"harness": "codex",
                           "env": {"OPENAI_API_KEY": connections._openai_compatible_api_key(key_config)}})
        return connection

    raise NotImplementedError("MVP limitation: Codex registry-model support currently only "
                              f"supports clembench models with backend='openrouter', backend='openai', "
                              "or backend='openai_compatible'. "
                              f"Model {clem_model!r} has backend={backend!r}.")
