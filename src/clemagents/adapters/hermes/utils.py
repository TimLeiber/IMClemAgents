"""Hermes-native observation hooks and terminal episode guards.

No hook changes prompts, tools, model output or request parameters. The only
interruptions are a finished game or an explicitly requested control that the
native harness did not send. This module is loaded through Hermes's plugin API.
"""

import dataclasses
import json
import os
import re
import subprocess
import threading
import time
from functools import partial
from pathlib import Path
from typing import Any

from ..utils import GAME_MCP_SERVER_NAME, read_game_completion, redact_sensitive
from ..utils import model_connection as connections

HOOKS = ("pre_api_request", "post_api_request", "api_request_error", "pre_tool_call", "post_tool_call",
         "subagent_start", "subagent_stop")
FIELDS = ("session_id", "task_id", "turn_id", "api_request_id", "api_call_count", "model", "provider", "base_url",
          "api_mode", "started_at", "ended_at", "request", "response", "assistant_message", "usage", "finish_reason",
          "tool_name", "tool_call_id", "args", "result", "status", "error", "error_type", "error_message",
          "status_code", "retry_count", "max_retries", "retryable", "reason", "duration_ms")
_lock = threading.Lock()


def _json_value(value):
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if hasattr(value, "__dict__"):
        return {key: item for key, item in vars(value).items() if not key.startswith("_")}
    return str(value)


def _write(event):
    path = os.environ.get("CLEM_HERMES_OBSERVER_PATH")
    if not path:
        raise SystemExit("Hermes observer output path is missing")
    line = redact_sensitive(json.dumps({"timestamp": time.time(), **event}, ensure_ascii=False, default=_json_value))
    try:
        with _lock:
            with Path(path).open("a", encoding="utf-8") as output:
                output.write(line + "\n")
                output.flush()
    except OSError as error:
        raise SystemExit(f"Hermes native observation could not be saved: {error}") from error


def _native_reasoning(body):
    # inspect a copy of native kwargs, including fields the sdk merges from extra_body
    fields = {**body, **(body.get("extra_body") or {})}
    return {key: fields[key] for key in ("reasoning", "reasoning_effort", "thinking", "chat_template_kwargs")
            if key in fields}


def _truncated(value):
    if isinstance(value, dict):
        return value.get("_truncated") is True or any(_truncated(item) for item in value.values())
    return isinstance(value, list) and any(_truncated(item) for item in value)


def observe(hook, **kwargs):
    """Persist a native event, without returning a transform or tool directive."""
    completion = os.environ.get("CLEM_HERMES_COMPLETION_PATH")
    if hook in {"pre_api_request", "pre_tool_call"} and completion and read_game_completion(Path(completion)):
        _write({"hook": "episode_closed", "content": "Stopped before additional work after game completion"})
        # systemexit bypasses observer exception suppression and unwinds native finalizers
        # the adapter also terminates the process group and exports the saved session
        raise SystemExit(0)

    event = {"hook": hook, **{key: kwargs[key] for key in FIELDS if key in kwargs}}
    if hook == "pre_api_request":
        snapshot = kwargs.get("request")
        snapshot = snapshot if isinstance(snapshot, dict) else {}
        native_body = snapshot.get("body")
        complete = isinstance(native_body, dict) and not _truncated(snapshot)
        body = dict(native_body) if isinstance(native_body, dict) else {}
        # the native sanitized snapshot caps long strings; raw messages are a documented hook field
        if isinstance(kwargs.get("request_messages"), list):
            body["input" if "input" in body else "messages"] = kwargs["request_messages"]
        event["request"] = {**snapshot, "body": body}
        effort = os.environ.get("CLEM_HERMES_REQUESTED_EFFORT")
        controls = _native_reasoning(body)
        event["native_reasoning_controls"] = controls
        event["reasoning_verification"] = "unverified_snapshot" if not complete else "observed" if controls else "absent"
        if effort and not complete:
            # an incomplete log cannot prove that the actual request omitted a setting
            event["capture_warning"] = ("Reasoning could not be verified from the incomplete native observer snapshot; "
                                        "the request was not blocked. Inspect the full native request dump.")
        if effort and complete and not controls:
            _write({**event, "hook": "unsupported_control", "requested_effort": effort,
                    "content": "Hermes omitted the requested reasoning control for this provider/model; no request sent"})
            raise SystemExit(2)

    if os.environ.get("CLEM_HERMES_TRACE_MODEL_IO") == "1":
        _write(event)


def register(ctx):
    """Register only native observers, never tools or prompt transforms."""
    for hook in HOOKS:
        ctx.register_hook(hook, partial(observe, hook))


def configure(home, config, observer_path, completion_path, effort, trace_model_io):
    """Write an isolated native Hermes config and observation plugin.

    Args:
        home: episode-specific Hermes home
        config: native Hermes configuration
        observer_path: append-only native event log
        completion_path: generic game completion marker
        effort: explicitly requested native reasoning effort, if any
        trace_model_io: whether to record model and tool content

    Returns:
        environment variables consumed by Hermes and the observation plugin
    """
    import yaml

    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    plugin = home / "plugins" / "clemagents-observer"
    plugin.mkdir(parents=True, exist_ok=True)
    (plugin / "plugin.yaml").write_text("name: clemagents-observer\nversion: '1'\ndescription: episode observation\n", encoding="utf-8")
    # use the same package source as the adapter even when hermes prepends site-packages
    source_root = str(Path(__file__).resolve().parents[3])
    loader = f"import sys\nsys.path.insert(0, {source_root!r})\nfrom clemagents.adapters.hermes.utils import register\n"
    (plugin / "__init__.py").write_text(loader, encoding="utf-8")
    config = {**config, "plugins": {"enabled": ["clemagents-observer"]}}
    path = home / "config.yaml"
    temporary = home / "config.yaml.tmp"
    temporary.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)
    return {"HERMES_HOME": str(home), "CLEM_HERMES_OBSERVER_PATH": str(observer_path),
            "CLEM_HERMES_COMPLETION_PATH": str(completion_path), "CLEM_HERMES_REQUESTED_EFFORT": effort or "",
            "CLEM_HERMES_TRACE_MODEL_IO": "1" if trace_model_io else "0",
            "HERMES_DUMP_REQUESTS": "1" if trace_model_io else "0"}


def check_hooks():
    """Verify installed native hook support without starting inference."""
    from hermes_cli.plugins import VALID_HOOKS, discover_plugins, get_plugin_manager
    from hermes_constants import VALID_REASONING_EFFORTS, parse_reasoning_effort

    effort = os.environ.get("CLEM_HERMES_REQUESTED_EFFORT", "")
    if effort and parse_reasoning_effort(effort) is None:
        allowed = ", ".join(["none", *sorted(VALID_REASONING_EFFORTS)])
        raise SystemExit(f"Hermes reasoning effort {effort!r} is not accepted by this installed version; use {allowed}")

    missing = set(HOOKS) - VALID_HOOKS
    if missing:
        raise SystemExit(f"Installed Hermes lacks required observation hooks: {sorted(missing)}")
    discover_plugins()
    plugins = get_plugin_manager().list_plugins()
    observer = next((item for item in plugins if item["name"] == "clemagents-observer"), {})
    if observer.get("error") or observer.get("hooks") != len(HOOKS):
        raise SystemExit(f"Hermes observation plugin did not load: {observer.get('error', 'missing hooks')}")
    _write({"hook": "observer_ready"})
    print("Hermes observation hooks ready")



def _resolve_model_connection(model_spec: dict[str, Any]) -> dict[str, Any]:
    backend = model_spec.get("backend")
    model_id = model_spec.get("model_id") or model_spec["model_name"]
    if backend == "openrouter":
        key_config = connections._openrouter_key_config()
        return {"harness": "hermes", "clem_model": model_spec["model_name"], "backend": backend,
                "provider": "openrouter", "model": model_id, "display_model": model_spec["model_name"],
                "base_url": connections._openrouter_openai_base_url(key_config),
                "env": {"OPENROUTER_API_KEY": connections._openrouter_api_key(key_config)}}
    if backend == "openai_compatible":
        key_config = connections._openai_compatible_key_config()
        connection = connections._openai_compatible_common(model_spec, key_config)
        connection.update({"harness": "hermes", "provider": "openai-api",
                           "env": {"OPENAI_BASE_URL": connection["base_url"],
                                   "OPENAI_API_KEY": connections._openai_compatible_api_key(key_config)}})
        return connection
    raise NotImplementedError(f"Hermes does not support registry backend {backend!r}")


def _check_setup(trace_parts):
    # mcp test prints errors with exit code zero, so require the actual inventory too
    commands = [["hermes", "mcp", "test", GAME_MCP_SERVER_NAME],
                ["python", "-m", "clemagents.adapters.hermes.utils"]]
    for command in commands:
        result = subprocess.run(command, text=True, capture_output=True, timeout=60)
        trace_parts.extend([" ".join(command), result.stdout, result.stderr])
        if result.returncode:
            detail = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else ""
            raise RuntimeError(f"{' '.join(command)} failed ({result.returncode}): {redact_sensitive(detail)}")
        if command[1:3] == ["mcp", "test"]:
            inventory = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout)
            if not all(re.search(rf"(?m)^\s+(?:mcp__game__)?{name}\s", inventory)
                       for name in ("start_game", "submit_response")):
                raise RuntimeError("Hermes MCP test did not discover start_game and submit_response; chat not started")


def _session_id(observer_path, text):
    if observer_path.exists():
        for line in observer_path.read_text(encoding="utf-8").splitlines():
            try:
                session = json.loads(line).get("session_id")
                if session:
                    return session
            except json.JSONDecodeError:
                pass
    match = re.search(r"(?:^Session:\s*|hermes --resume\s+|\bsession=)([A-Za-z0-9_.:-]+)", text, re.MULTILINE)
    return match.group(1) if match else None


def _export_session(session, run_dir, artifacts, trace_parts):
    path = run_dir / "hermes_session_export.jsonl"
    result = subprocess.run(["hermes", "sessions", "export", "--session-id", session, str(path)],
                            text=True, capture_output=True, timeout=60)
    trace_parts.extend(["hermes_sessions_export:", result.stdout, result.stderr])
    if path.exists():
        artifacts["hermes_session_export"] = path


if __name__ == "__main__":
    check_hooks()
