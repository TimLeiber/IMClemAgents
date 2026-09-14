import json
import os
import tempfile
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any

from clemagents import adapters
from clemagents.adapters import harness_class_for_agent
from clemagents.adapters.base import AgentRunResult

CONFIG_PATH = (Path(adapters.__file__).resolve().parent / "utils" / "external_agent_config.yaml")


def _write_artifact_marker(payload: dict[str, Any], environment_variable: str) -> None:
    """Atomically publish an artifact lifecycle marker."""

    marker_value = os.environ.get(environment_variable)

    if not marker_value:
        return

    marker_path = Path(marker_value)
    marker_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=marker_path.parent, prefix=f".{marker_path.name}.",
                                     suffix=".tmp", delete=False) as marker_file:
        json.dump(payload, marker_file, ensure_ascii=False)
        marker_file.flush()
        temporary_path = Path(marker_file.name)

    temporary_path.replace(marker_path)


def run_external_agent_episode(agent_name: str, registry_path: str | Path, output_root: str | Path | None,
                               instruction: str | None = None,
                               run_metadata: dict[str, Any] | None = None) -> AgentRunResult:
    """Run one episode with an external agent.

    Args:
        agent_name: name of the agent in the agent registry
        registry_path: path to the external-agent registry
        output_root: directory for run artifacts or None to disable output
        instruction: instruction passed to the agent; loads the shared
            meta prompt when omitted
        run_metadata: optional metadata included in the run summary

    Returns:
        the result returned by the external-agent adapter
    """

    registry_path = Path(registry_path).expanduser()
    output_dir = None

    # load the shared meta prompt when no instruction was provided
    if instruction is None:
        if not CONFIG_PATH.exists():
            raise FileNotFoundError(f"Missing external-agent configuration: {CONFIG_PATH}")

        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

        if not isinstance(config, dict):
            raise ValueError("External-agent configuration must be a mapping: "
                             f"{CONFIG_PATH}")

        instruction = config.get("meta_prompt")

        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Missing non-empty meta_prompt in configuration: "
                             f"{CONFIG_PATH}")

    # create a timestamped output directory when output is enabled
    if output_root is not None:
        output_root = Path(output_root).expanduser()
        # include microseconds so rapid failures cannot reuse an earlier
        # episode's directory and leak native artifacts across instances
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_dir = output_root / agent_name / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        # publish before adapter initialization so interrupted runs remain recoverable
        marker_value = os.environ.get("AGENT_ARTIFACTS_STARTED_PATH")
        if marker_value:
            relative_path = output_dir.resolve().relative_to(Path(marker_value).resolve().parent)
            _write_artifact_marker({"artifact_directory": str(relative_path)}, "AGENT_ARTIFACTS_STARTED_PATH")

    # load the requested agent specification from the registry
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    matches = [entry for entry in registry if entry["agent_name"] == agent_name]

    if not matches:
        known_agents = [entry["agent_name"] for entry in registry]
        raise ValueError(f"Unknown agent '{agent_name}'. Known agents: {known_agents}")

    spec = matches[0]
    # load the actual agent and store it in this variable
    harness_class = harness_class_for_agent(agent_name, registry_path)
    agent = harness_class(**spec.get("agent_config", {}))
    print("agent_loop_instruction_start")
    print(instruction)
    print("agent_loop_instruction_end")
    result = agent.run_episode(instruction=instruction, output_dir=output_dir)

    # write the container-local artifacts used for diagnostics and summaries
    if output_dir is not None:
        result.metadata["artifact_directory"] = str(output_dir)
        trace_parts = ["agent_loop_instruction_start", instruction, "agent_loop_instruction_end"]
        adapter_messages = result.artifacts.get("adapter_messages")

        if adapter_messages is not None:
            adapter_trace_path = Path(adapter_messages)

            if adapter_trace_path.exists():
                trace_parts.append(adapter_trace_path.read_text(encoding="utf-8"))

        trace_path = output_dir / "agent_trace.log"
        trace_path.write_text("\n".join(trace_parts), encoding="utf-8")
        result.artifacts["agent_trace"] = trace_path

        # write a machine-readable summary alongside the agent artifacts
        summary = {"agent_name": agent_name,
                   "registry_path": str(registry_path),
                   "output_dir": str(output_dir),
                   "success": result.success,
                   "metadata": result.metadata,
                   "artifacts": {key: str(value) for key, value in result.artifacts.items()},
                   "run_metadata": run_metadata or {}}

        summary_path = output_dir / "run_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        result.artifacts["run_summary"] = summary_path

        print(f"Wrote run summary to {summary_path}")

    return result


if __name__ == "__main__":
    ready_payload: dict[str, Any] = {"ready": False, "status": "error"}

    try:
        result = run_external_agent_episode(agent_name=os.environ["AGENT_NAME"],
                                            registry_path="/tmp/agent_registry.json",
                                            output_root=os.environ.get("AGENT_ARTIFACT_ROOT"))

        print("success:", result.success)

        for name, path in result.artifacts.items():
            print(f"{name}: {path}")

        ready_payload = {"ready": True, "status": "complete", "success": result.success}
        artifact_directory = result.metadata.get("artifact_directory")
        marker_value = os.environ.get("AGENT_ARTIFACTS_READY_PATH")

        if isinstance(artifact_directory, str) and marker_value:
            try:
                ready_payload["artifact_directory"] = str(Path(artifact_directory).resolve().relative_to(Path(marker_value).resolve().parent))
            except ValueError:
                # never expose an arbitrary absolute path to the host copier
                ready_payload["artifact_error"] = ("artifact directory is outside the shared episode directory")
    except BaseException as error:
        ready_payload["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        # this is deliberately the final persistence action in the container
        # forced shutdown recovers the started directory without claiming finalization
        _write_artifact_marker(ready_payload, "AGENT_ARTIFACTS_READY_PATH")
