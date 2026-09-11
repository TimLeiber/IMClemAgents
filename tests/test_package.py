import importlib
import json
import subprocess
import sys
import types
from unittest.mock import patch
import pytest
from importlib.resources import files
from pathlib import Path

from clemagents.adapters import harness_class_for_agent
from clemagents.adapters.base import AgentRunResult, ExternalAgentHarness
from clemagents.adapters.model_connection import resolve_agent_model_connection


def test_installed_resources_are_available():
    package = files("clemagents")
    for name in ("adapters/external_agent_config.yaml", "mcp/mcp_server_config.yaml", "docker/agent-sandbox/Dockerfile",
                 "docker/agent-sandbox/run_agent_container.py"):
        assert package.joinpath(name).is_file()


def test_commands_work_outside_checkout(tmp_path):
    for module in ("clemagents.run_pipeline", "clemagents.transcribe_agent_loop"):
        result = subprocess.run([sys.executable, "-m", module, "--help"], cwd=tmp_path, text=True, capture_output=True,
                                timeout=60)
        assert result.returncode == 0, result.stderr
        assert "--results_dir" in result.stdout


@pytest.mark.parametrize("effort", ["low", "none", {"native": "custom-setting"}])
def test_new_adapter_needs_no_central_dispatch_change(tmp_path, monkeypatch, effort):
    name = "clemagents.adapters.fixture"
    module = types.ModuleType(name)

    class FixtureHarness(ExternalAgentHarness):

        @classmethod
        def resolve_model_connection(cls, model_spec, agent_config):
            assert model_spec == {"model_name": "fixture-model", "backend": "fixture"}
            assert agent_config["reasoning_effort"] == effort
            return {"model": model_spec["model_name"], "backend": "fixture"}

        def run_episode(self, instruction, output_dir=None):
            return AgentRunResult(success=True)

    FixtureHarness.__module__ = name
    module.FixtureHarness = FixtureHarness
    monkeypatch.setitem(sys.modules, name, module)
    registry = tmp_path / "agent_registry.json"
    registry.write_text(json.dumps([{"agent_name": "fixture-agent",
                                     "backend": "fixture",
                                     "agent_config": {"clem_model": "fixture-model",
                                                      "reasoning_effort": effort}}]))

    assert harness_class_for_agent("fixture-agent", registry) is FixtureHarness
    with patch("clemagents.adapters.model_connection._find_model_spec", return_value={"model_name": "fixture-model",
                                                                                      "backend": "fixture"}):
        assert resolve_agent_model_connection("fixture-agent", registry) == {"model": "fixture-model",
                                                                             "backend": "fixture"}


def test_core_is_an_external_dependency():
    core = importlib.import_module("clemcore")
    package = importlib.import_module("clemagents")
    assert Path(core.__file__).resolve().parent != Path(package.__file__).resolve().parent
    assert importlib.util.find_spec("clemcore.agents") is None


def test_direct_model_skips_registry_resolution(tmp_path):
    registry = tmp_path / "agent_registry.json"
    registry.write_text(json.dumps([{"agent_name": "direct",
                                     "backend": "not-installed",
                                     "agent_config": {"model": "native-model",
                                                      "reasoning_effort": "custom"}}]))
    with patch("clemagents.adapters.model_connection._find_model_spec") as lookup:
        assert resolve_agent_model_connection("direct", registry) is None
        lookup.assert_not_called()


@pytest.mark.parametrize("backend", ["openrouter", "openai_compatible"])
@pytest.mark.parametrize("harness", ["codex", "claude_code", "hermes", "openclaw"])
@pytest.mark.parametrize("effort", [None, "low", "none", "off", " HIGH "])
def test_model_resolution_does_not_construct_request_overrides(backend, harness, effort):
    from copy import deepcopy
    from clemagents.adapters import model_connection

    spec = {"model_name": "test-model",
            "model_id": "provider/test-model",
            "backend": backend,
            "context_size": "128k",
            "model_config": {"extra_body": {"reasoning": {"enabled": True,
                                                          "effort": "max",
                                                          "summary": "auto"},
                                            "custom_field": {"keep": True}}}}
    original = deepcopy(spec)
    config = {"clem_model": "test-model", "reasoning_effort": effort}
    original_config = deepcopy(config)
    module = importlib.import_module(f"clemagents.adapters.{harness}")
    adapter = next(value for value in vars(module).values() if isinstance(value, type)
                   and value.__module__ == module.__name__ and issubclass(value, ExternalAgentHarness))
    key_config = {"api_key": "fixture-key", "base_url": "http://localhost:11434/v1"}
    with patch.object(model_connection, f"_{backend}_key_config", return_value=key_config):
        baseline = adapter.resolve_model_connection(spec, {})
        actual = adapter.resolve_model_connection(spec, config)

    assert actual == baseline
    assert "request_body_overrides" not in actual
    assert "generation_overrides" not in actual
    assert "upstream_model" not in actual
    assert spec == original
    assert config == original_config
