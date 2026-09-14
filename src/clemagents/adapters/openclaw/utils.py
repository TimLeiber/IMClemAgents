"""Native openclaw configuration and execution helpers."""

import json
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from ..utils import model_connection as connections
from ..utils import redact_sensitive


def _validate_openclaw_model_connection(connection: dict[str, Any] | None) -> None:
    """Validate the native OpenRouter model and credential expected by OpenClaw."""
    if not connection or connection.get("backend") != "openrouter":
        return

    runtime_model = connection.get("model")
    if not isinstance(runtime_model, str) or not runtime_model.startswith("openrouter/"):
        raise ValueError("OpenClaw OpenRouter connections require a canonical "
                         f"openrouter/<provider>/<model> reference, got {runtime_model!r}.")

    environment = connection.get("env")
    if not isinstance(environment, dict) or not environment.get("OPENROUTER_API_KEY"):
        raise ValueError("OpenClaw OpenRouter connections require OPENROUTER_API_KEY.")


def _error_detail(result: subprocess.CompletedProcess[str]) -> str | None:
    """Return a short, redacted error from a failed OpenClaw command."""
    try:
        payload = json.loads(result.stdout)
        detail = payload.get("error", {}).get("message")
        if isinstance(detail, str) and detail.strip():
            return redact_sensitive(detail)
    except (ValueError, AttributeError):
        pass
    lines = [line.strip() for output in (result.stderr, result.stdout) for line in output.splitlines() if line.strip()]
    return redact_sensitive(lines[-1]) if lines else None


@contextmanager
def _openclaw_gateway(base_command: list[str], log_path: Path, startup_timeout: float = 45):
    """Keep the native Gateway alive for the episode and stop it before export."""
    command = base_command + ["gateway", "run"]
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + startup_timeout
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"OpenClaw Gateway exited during startup ({process.returncode}); "
                                       f"see {log_path.name}")
                try:
                    with urlopen("http://127.0.0.1:18789/startupz", timeout=1) as response:
                        if response.status == 200:
                            break
                except (URLError, TimeoutError):
                    pass
                time.sleep(0.1)
            else:
                raise RuntimeError(f"OpenClaw Gateway was not ready after {startup_timeout}s; "
                                   f"see {log_path.name}")
            yield
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def _resolve_model_connection(model_spec: dict[str, Any]) -> dict[str, Any]:
    clem_model = model_spec["model_name"]
    backend = model_spec.get("backend")
    model_id = model_spec.get("model_id") or model_spec.get("model_name")

    if backend == "openrouter":
        key_config = connections._openrouter_key_config()

        return {"harness": "openclaw",
                "clem_model": model_spec["model_name"],
                "backend": "openrouter",
                "model": f"openrouter/{model_id}",
                "display_model": model_spec["model_name"],
                "base_url": connections._openrouter_openai_base_url(key_config),
                "openclaw_provider": "openrouter",
                "env": {"OPENROUTER_API_KEY": connections._openrouter_api_key(key_config)}}

    if backend == "openai_compatible":
        key_config = connections._openai_compatible_key_config()
        provider_name = "openai_compatible"
        api_key_env = "OPENAI_COMPATIBLE_API_KEY"
        runtime_model = f"{provider_name}/{model_id}"
        common = connections._openai_compatible_common(model_spec, key_config)

        model_definition: dict[str, Any] = {"id": model_id,
                                            "name": model_spec["model_name"],
                                            "api": "openai-completions",
                                            "reasoning": connections._model_reasoning_enabled(model_spec),
                                            "input": _openclaw_input_modalities(model_spec)}

        model_config = model_spec.get("model_config") or {}
        extra_body = {}

        if isinstance(model_config, dict):
            extra_body = model_config.get("extra_body") or {}

        chat_template_kwargs = {}

        if isinstance(extra_body, dict):
            raw_chat_template_kwargs = extra_body.get("chat_template_kwargs") or {}

            if isinstance(raw_chat_template_kwargs, dict):
                chat_template_kwargs = raw_chat_template_kwargs

        if "enable_thinking" in chat_template_kwargs:
            # declare the native protocol capability without fixing its on/off value
            # the agent's thinking option controls each run
            model_definition["reasoning"] = True
            model_definition["compat"] = {"thinkingFormat": "qwen-chat-template"}

        openclaw_model_config: dict[str, Any] = {"alias": model_spec["model_name"]}

        context_window = connections._model_context_window(model_spec)

        if context_window is not None:
            model_definition["contextWindow"] = context_window

        model_definition["maxTokens"] = 8192

        return {**common, "harness": "openclaw",
                "runtime_model": runtime_model,
                "openclaw_provider": provider_name,
                "env": {api_key_env: connections._openai_compatible_api_key(key_config)},
                "openclaw_config_patch": {"models": {"mode": "merge",
                                                     "providers": {provider_name: {"baseUrl": common["base_url"],
                                                                                   "apiKey": f"${{{api_key_env}}}",
                                                                                   "api": "openai-completions",
                                                                                   "models": [model_definition]}}},
                                          "agents": {"defaults": {"models": {runtime_model: openclaw_model_config}}}}}

    raise NotImplementedError("MVP limitation: OpenClaw registry-model support currently only "
                              f"supports clembench models with backend='openrouter' or backend='openai_compatible'. "
                              f"Model {clem_model!r} has backend={backend!r}.")


def _openclaw_input_modalities(model_spec: dict[str, Any]) -> list[str]:
    """Translate registry input capabilities into OpenClaw's model schema."""
    model_config = model_spec.get("model_config") or {}

    if not isinstance(model_config, dict):
        return ["text"]

    configured = model_config.get("input_modalities")

    if not isinstance(configured, list):
        return ["text"]

    modalities = ["text"]

    for modality in configured:
        if not isinstance(modality, str):
            continue

        normalized = modality.strip().lower()

        if normalized == "image" and normalized not in modalities:
            modalities.append(normalized)

    return modalities
