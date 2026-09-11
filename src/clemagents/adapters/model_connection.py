import json
import os
from pathlib import Path
from typing import Any

from clemcore.backends import ModelRegistry


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _model_spec_to_dict(model_spec: Any) -> dict[str, Any]:
    if isinstance(model_spec, dict):
        return dict(model_spec)

    return {
        "model_name": getattr(model_spec, "model_name", None),
        "model_id": getattr(model_spec, "model_id", None),
        "backend": getattr(model_spec, "backend", None),
        "context_size": getattr(model_spec, "context_size", None),
        "model_config": getattr(model_spec, "model_config", None),
    }


def _find_model_spec(clem_model: str) -> dict[str, Any]:
    registry = ModelRegistry.from_packaged_and_cwd_files()

    for model_spec in registry:
        model_spec_dict = _model_spec_to_dict(model_spec)

        if model_spec_dict.get("model_name") == clem_model:
            return model_spec_dict

    raise KeyError(
        f"Could not find clem_model={clem_model!r}. "
        "Run this from the clembench directory so model_registry.json is discoverable."
    )


def _load_key_json() -> dict[str, Any]:
    candidates = []

    if os.environ.get("CLEM_KEY_FILE"):
        candidates.append(Path(os.environ["CLEM_KEY_FILE"]).expanduser())

    candidates.extend([
        Path.cwd() / "key.json",
        Path.home() / ".clemcore" / "key.json",
    ])

    merged = {}

    for path in reversed(candidates):
        if path.exists():
            data = _read_json(path)

            if isinstance(data, dict):
                merged.update(data)

    return merged


def _openrouter_key_config() -> dict[str, Any]:
    key_json = _load_key_json()
    config = key_json.get("openrouter", {})

    if not isinstance(config, dict):
        return {}

    return config


def _openrouter_api_key(config: dict[str, Any]) -> str:
    api_key = config.get("api_key") or os.environ.get("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError(
            "No OpenRouter API key found. Expected clembench/key.json entry "
            "'openrouter.api_key' or env var OPENROUTER_API_KEY."
        )

    return str(api_key)


def _openai_key_config() -> dict[str, Any]:
    key_json = _load_key_json()
    config = key_json.get("openai", {})

    if not isinstance(config, dict):
        return {}

    return config


def _openai_api_key(config: dict[str, Any]) -> str:
    api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "No OpenAI API key found. Expected clembench/key.json entry "
            "'openai.api_key' or env var OPENAI_API_KEY."
        )

    return str(api_key)


def _openai_compatible_key_config() -> dict[str, Any]:
    key_json = _load_key_json()
    config = key_json.get("openai_compatible", {})

    if not isinstance(config, dict):
        return {}

    return config


def _openai_compatible_api_key(config: dict[str, Any]) -> str:
    api_key = (
        config.get("api_key")
        or os.environ.get("OPENAI_COMPATIBLE_API_KEY")
    )

    if not api_key:
        raise RuntimeError(
            "No OpenAI-compatible API key found. Expected clembench/key.json entry "
            "'openai_compatible.api_key' or env var OPENAI_COMPATIBLE_API_KEY."
        )

    return str(api_key)


def _openai_compatible_base_url(config: dict[str, Any]) -> str:
    base_url = config.get("base_url") or os.environ.get("OPENAI_COMPATIBLE_BASE_URL")

    if not base_url:
        raise RuntimeError(
            "No OpenAI-compatible base URL found. Expected clembench/key.json entry "
            "'openai_compatible.base_url' or env var OPENAI_COMPATIBLE_BASE_URL."
        )

    return str(base_url).rstrip("/")


def _model_context_window(model_spec: dict[str, Any]) -> int | None:
    context_size = model_spec.get("context_size")

    if context_size is None:
        return None

    if isinstance(context_size, int):
        return context_size

    text = str(context_size).strip().lower().replace(",", "")

    if text.endswith("k") and text[:-1].replace(".", "", 1).isdigit():
        return int(float(text[:-1]) * 1000)

    if text.isdigit():
        return int(text)

    return None


def _model_reasoning_enabled(model_spec: dict[str, Any]) -> bool:
    model_config = model_spec.get("model_config") or {}

    if not isinstance(model_config, dict):
        return False

    extra_body = model_config.get("extra_body") or {}

    if not isinstance(extra_body, dict):
        return False

    chat_template_kwargs = extra_body.get("chat_template_kwargs") or {}

    if isinstance(chat_template_kwargs, dict):
        enable_thinking = chat_template_kwargs.get("enable_thinking")

        if isinstance(enable_thinking, bool):
            return enable_thinking

    reasoning = extra_body.get("reasoning") or {}

    if isinstance(reasoning, dict):
        enabled = reasoning.get("enabled")

        if isinstance(enabled, bool):
            return enabled

        effort = reasoning.get("effort")

        if effort == "none":
            return False

        if effort is not None:
            return True

    return False


def _model_request_body_overrides(model_spec: dict[str, Any]) -> dict[str, Any]:
    model_config = model_spec.get("model_config") or {}

    if not isinstance(model_config, dict):
        return {}

    extra_body = model_config.get("extra_body") or {}
    return dict(extra_body) if isinstance(extra_body, dict) else {}


def _openai_compatible_verify_tls(config: dict[str, Any]) -> bool:
    configured = config.get("verify_tls")

    if isinstance(configured, bool):
        return configured

    return True


def _openai_compatible_common(
    model_spec: dict[str, Any],
    key_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "clem_model": model_spec["model_name"],
        "backend": "openai_compatible",
        "model": model_spec.get("model_id") or model_spec.get("model_name"),
        "display_model": model_spec["model_name"],
        "base_url": _openai_compatible_base_url(key_config),
        "verify_tls": _openai_compatible_verify_tls(key_config),
        "request_body_overrides": _model_request_body_overrides(model_spec),
    }


def _openrouter_openai_base_url(config: dict[str, Any]) -> str:
    base_url = str(config.get("base_url") or "https://openrouter.ai/api/v1").rstrip("/")

    # OpenAI-compatible clients need OpenRouter's OpenAI-compatible endpoint.
    if base_url.endswith("/api") and not base_url.endswith("/api/v1"):
        return f"{base_url}/v1"

    return base_url


def resolve_agent_model_connection(agent_name: str,
                                   registry_path: str | Path) -> dict[str, Any] | None:
    registry_path = Path(registry_path)
    registry = _read_json(registry_path)

    if isinstance(registry, dict):
        registry = [registry]

    for entry in registry:
        if entry.get("agent_name") != agent_name:
            continue

        agent_config = entry.get("agent_config", {})
        clem_model = agent_config.get("clem_model")

        if not clem_model:
            return None

        from . import harness_class_for_agent

        adapter = harness_class_for_agent(agent_name, registry_path)
        connection = adapter.resolve_model_connection(clem_model)

        # A reasoning setting declared in the agent registry takes precedence
        # over the model registry's extra_body. ``none`` and ``off`` disable
        # reasoning; other non-empty values select an enabled effort.
        # The value is carried as OpenAI-compatible request_body_overrides,
        # which the generic compatibility proxy merges into forwarded
        # OpenAI-shaped requests (Anthropic Messages requests are exempt).
        effort = agent_config.get("reasoning_effort")

        if isinstance(connection, dict) and isinstance(effort, str) and effort.strip():
            overrides = connection.get("request_body_overrides")

            if not isinstance(overrides, dict):
                overrides = {}

            reasoning = overrides.get("reasoning")

            if not isinstance(reasoning, dict):
                reasoning = {}

            normalized_effort = effort.strip().lower()

            if normalized_effort in {"none", "off"}:
                reasoning = {**reasoning, "enabled": False}
                reasoning.pop("effort", None)
            else:
                reasoning = {
                    **reasoning,
                    "enabled": True,
                    "effort": normalized_effort,
                }
            overrides = {**overrides, "reasoning": reasoning}
            connection["request_body_overrides"] = overrides

        return connection

    raise KeyError(f"Could not find agent {agent_name!r} in {registry_path}.")
