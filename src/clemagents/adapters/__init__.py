import importlib
import inspect
import json
from pathlib import Path

from clemagents.adapters.base import ExternalAgentHarness


def harness_class_for_agent(agent_name: str, registry_path: str | Path) -> type[ExternalAgentHarness]:
    """Load the single harness class from adapters/<backend>/<backend>.py.

    Args:
        agent_name: configured agent selected for the episode
        registry_path: JSON file containing agent configurations

    Returns:
        the adapter class, without a central list of supported harnesses
    """

    registry_path = Path(registry_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    matches = [entry for entry in registry if entry.get("agent_name") == agent_name]

    if not matches:
        known_agents = [entry.get("agent_name") for entry in registry]
        raise ValueError(f"Unknown agent '{agent_name}'. Known agents: {known_agents}")

    backend = matches[0].get("backend")

    if not isinstance(backend, str) or not backend:
        raise ValueError(f"Agent '{agent_name}' has no valid backend")

    module = importlib.import_module(f"clemagents.adapters.{backend}.{backend}")
    harness_classes = [candidate for candidate in vars(module).values()
                       if inspect.isclass(candidate) and candidate is not ExternalAgentHarness
                       and issubclass(candidate, ExternalAgentHarness) and candidate.__module__ == module.__name__]

    if len(harness_classes) != 1:
        raise RuntimeError(f"Expected exactly one external-agent harness in {module.__name__}, "
                           f"found {len(harness_classes)}")

    return harness_classes[0]
