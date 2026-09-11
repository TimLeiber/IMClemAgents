"""Adapter skeleton: implement the two native operations before running it."""

import json
from pathlib import Path

from clemagents.adapters.base import AgentRunResult, ExternalAgentHarness
from clemagents.adapters.utils import mcp_environment


class ExampleHarness(ExternalAgentHarness):
    def __init__(self, model: str, mcp_url: str = "http://localhost:8001/mcp"):
        self.model = model
        self.mcp_url = mcp_url

    def run_episode(self, instruction: str, output_dir=None) -> AgentRunResult:
        # Configure the native harness to launch this MCP server:
        command = ["python", "-m", "clemagents.mcp.bridge"]
        environment = mcp_environment(self.mcp_url)
        # Run its documented CLI/SDK with instruction and self.model. Save
        # native artifacts in output_dir, then return AgentRunResult with paths.
        raise NotImplementedError("Implement native harness setup and execution")

    @classmethod
    def parse_agent_trace(cls, episode_dir: Path, metadata=None) -> dict:
        # This example assumes the harness already wrote uniform events. For
        # another format, parse its native logs here into the same event fields.
        path = episode_dir / "example_events.json"
        if not path.exists():
            events = [{"sequence": 1, "type": "trace_warning",
                       "content": "Native event capture is missing"}]
        else:
            events = json.loads(path.read_text(encoding="utf-8"))
        return {"schema_version": 1, "backend": "example", "events": events}
