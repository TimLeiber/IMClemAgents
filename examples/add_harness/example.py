"""Adapter skeleton with native execution left for the contributor to implement."""

import json
from pathlib import Path

from clemagents.adapters.base import AgentRunResult, ExternalAgentHarness
from clemagents.adapters.utils.parse import missing_agent_trace
from clemagents.adapters.utils import mcp_environment


class ExampleHarness(ExternalAgentHarness):

    def __init__(self, model: str, mcp_url: str = "http://localhost:8001/mcp"):
        self.model = model
        self.mcp_url = mcp_url

    def run_episode(self, instruction: str, output_dir=None) -> AgentRunResult:
        # configure the native harness to launch this mcp server
        command = ["python", "-m", "clemagents.mcp.bridge"]
        environment = mcp_environment(self.mcp_url)
        # run the native cli or sdk with the instruction and selected model
        # save artifacts in output_dir and return their paths in the result
        raise NotImplementedError("Implement native harness setup and execution")

    @classmethod
    def parse_agent_trace(cls, episode_dir: Path, metadata=None) -> dict:
        # this example assumes the harness already wrote uniform events. for
        # another format, parse its native logs here into the same event fields
        path = episode_dir / "example_events.json"
        if not path.exists():
            return missing_agent_trace("Native event capture is missing", "example")
        events = json.loads(path.read_text(encoding="utf-8"))
        # the common serializer supplies version, backend and sequence numbers
        return {"events": events}
