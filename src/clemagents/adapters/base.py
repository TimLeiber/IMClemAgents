import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AgentRunResult:
    """Store the standardized result returned by an agent harness.

    Attributes:
        success: whether the harness completed the episode successfully
        artifacts: files and scalar values produced during the run
        metadata: structured details describing the run
    """

    success: bool
    artifacts: dict[str, Path | str | int | float | bool | None] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class ExternalAgentHarness(ABC):
    """Define the interface implemented by every external-agent harness."""

    @classmethod
    def resolve_model_connection(cls, clem_model: str) -> dict[str, Any]:
        """Translate a registry model into this harness's native configuration.

        Called on the host before container startup, only when agent_config
        includes clem_model. Adapters using their own credentials and a direct
        model identifier can omit this method.
        """
        raise NotImplementedError(f"{cls.__name__} does not support clem_model")

    @classmethod
    def parse_agent_trace(cls,
                          episode_dir: Path,
                          metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Parse native harness artifacts into the common agent-loop schema.

        Args:
            episode_dir: directory containing the artifacts for one episode
            metadata: optional metadata describing the agent run

        Returns:
            an empty trace until the harness provides its own parser
        """

        return {}

    @classmethod
    def serialize_standardized_agent_trace(cls,
                                           episode_dir: Path,
                                           artifact_dir: Path | None = None,
                                           metadata: dict[str, Any] | None = None) -> Path:
        """Serialize one harness's standardized agent trace.

        Args:
            episode_dir: directory containing the artifacts for one episode
            artifact_dir: optional finalized native-artifact directory used as
                the parser input while serialization remains in ``episode_dir``
            metadata: optional metadata describing the agent run

        Returns:
            path to the serialized standardized trace
        """

        agent_trace = cls.parse_agent_trace(episode_dir=artifact_dir or episode_dir,
                                            metadata=metadata)

        if not isinstance(agent_trace, dict):
            raise TypeError("Agent-trace parsers must return a dictionary")

        output_path = episode_dir / "agent_loop.json"
        output_path.write_text(
            json.dumps(agent_trace, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8"
        )

        return output_path

    @abstractmethod
    def run_episode(self,
                    instruction: str,
                    output_dir: Path | str | None = None) -> AgentRunResult:
        """Run one agent episode.

        Args:
            instruction: task instruction passed to the external agent
            output_dir: optional directory for adapter artifacts

        Returns:
            the standardized result of the agent run
        """

        raise NotImplementedError
