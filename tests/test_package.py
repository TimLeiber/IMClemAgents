import importlib
import json
import subprocess
import sys
import types
from importlib.resources import files
from pathlib import Path

from clemagents.adapters import harness_class_for_agent
from clemagents.adapters.base import AgentRunResult, ExternalAgentHarness
from clemagents.adapters.model_connection import resolve_agent_model_connection


def test_installed_resources_are_available():
    package = files("clemagents")
    for name in ("adapters/external_agent_config.yaml", "mcp/mcp_server_config.yaml",
                 "docker/agent-sandbox/Dockerfile", "docker/agent-sandbox/run_agent_container.py"):
        assert package.joinpath(name).is_file()


def test_commands_work_outside_checkout(tmp_path):
    for module in ("clemagents.run_pipeline", "clemagents.transcribe_agent_loop"):
        result = subprocess.run([sys.executable, "-m", module, "--help"], cwd=tmp_path,
                                text=True, capture_output=True, timeout=60)
        assert result.returncode == 0, result.stderr
        assert "--results_dir" in result.stdout


def test_new_adapter_needs_no_central_dispatch_change(tmp_path, monkeypatch):
    name = "clemagents.adapters.fixture"
    module = types.ModuleType(name)

    class FixtureHarness(ExternalAgentHarness):
        @classmethod
        def resolve_model_connection(cls, clem_model):
            return {"model": clem_model, "backend": "fixture"}

        def run_episode(self, instruction, output_dir=None):
            return AgentRunResult(success=True)

    FixtureHarness.__module__ = name
    module.FixtureHarness = FixtureHarness
    monkeypatch.setitem(sys.modules, name, module)
    registry = tmp_path / "agent_registry.json"
    registry.write_text(json.dumps([{
        "agent_name": "fixture-agent", "backend": "fixture",
        "agent_config": {"clem_model": "fixture-model"},
    }]))

    assert harness_class_for_agent("fixture-agent", registry) is FixtureHarness
    assert resolve_agent_model_connection("fixture-agent", registry) == {
        "model": "fixture-model", "backend": "fixture",
    }


def test_core_is_an_external_dependency():
    core = importlib.import_module("clemcore")
    package = importlib.import_module("clemagents")
    assert Path(core.__file__).resolve().parent != Path(package.__file__).resolve().parent
    assert importlib.util.find_spec("clemcore.agents") is None
