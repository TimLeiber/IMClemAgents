# Add a harness

Addings a harness requires you to first create a directory for it in the following way:
Create `src/clemagents/adapters/example/`, add an empty `__init__.py`, and copy
[example.py](example.py) into it as well. Follow [base.py](../../src/clemagents/adapters/base.py)
When defining the required methods for the adapter to interface with the engine.

```text
example/
    __init__.py
    example.py     # one ExternalAgentHarness subclass
    utils.py       # optional native configuration and execution helpers
    parse.py       # optional native artifact parser
```

In the games repository's `agent_registry.json`, you define which harness a model
uses by setting `backend`. For example, `"backend": "example"` loads the harness
class from `src/clemagents/adapters/example/example.py`. The directory and Python
file must both match this name, and the file must define one subclass of
`ExternalAgentHarness`.

This is similar to selecting a backend in regular Clembench, but here `backend`
selects the harness, not the model provider. See the existing entries in the
[agent registry](https://github.com/TimLeiber/IMClemAgents-clembench/blob/main/agent_registry.json)
for examples.

## Implement the class

| Method | Responsibility |
| --- | --- |
| `run_episode(instruction, output_dir)` | Configure and run the native harness; save artifacts and return `AgentRunResult` |
| `parse_agent_trace(episode_dir, metadata)` | Convert saved artifacts into ordered trace events |
| `resolve_model_connection(model_spec, agent_config)` | Required when using `clem_model`; optional for direct model configuration |

Expose supported native settings as constructor arguments. The adapter owns model
selection, credentials and reasoning controls. For `clem_model`, the resolver
returns a JSON-serializable connection dictionary that the adapter reads with
`load_model_connection` from `clemagents.adapters.utils`. Shared code does not
rewrite provider request bodies.

### Connect the harness to the game

The harness does not know the game tools yet. Your adapter must connect it to
the program that provides `start_game` and `submit_response`. That program is
already part of the framework: `clemagents.mcp.bridge`. It receives tool calls
from the harness and forwards them to the game running on the host.

Inside `run_episode`, use the harness's MCP settings to tell it to start this
program with the following command:

```bash
python -m clemagents.mcp.bridge
```

Here, `python` starts the Python interpreter inside the container. `-m` tells
Python to run the installed module named `clemagents.mcp.bridge`. The harness
starts it as a separate process when connecting to its MCP tools. This is not
a command for the model to generate or for the user to run manually.

For example, with the Claude Code SDK, the relevant configuration can be written
as follows inside `run_episode`:

```python
from claude_agent_sdk import ClaudeAgentOptions
from clemagents.adapters.utils import mcp_environment

options = ClaudeAgentOptions(mcp_servers={"game": {
    "type": "stdio",
    "command": "python",
    "args": ["-m", "clemagents.mcp.bridge"],
    "env": mcp_environment(self.mcp_url)}})
```

`command` and `args` are the SDK's way of storing the command shown above.
`stdio` means the harness sends and receives MCP messages through the process's
standard input and output. `env` supplies settings to that process.
`mcp_environment(self.mcp_url)` prepares those settings, including the host game
server's address and the experiment and instance selected by the pipeline.

Pass these options when starting the SDK. For another harness, register the same
command through that harness's MCP configuration format. The bridge and game
tools do not need to be reimplemented. The
[Claude Code adapter](../../src/clemagents/adapters/claude_code/claude_code.py)
shows the complete SDK setup. The [pipeline diagram](../../documentation/pipeline.md#mcp-interface)
shows how the tool call reaches the game and how feedback returns.

### Stop execution and save output

When the game finishes, the bridge writes a completion file. CLI adapters can
use `run_process_until_game_complete` to watch that file and stop the harness.
SDK adapters can check it with `read_game_completion` while receiving messages,
as the Claude Code adapter does.

Save the harness's logs and other output under `output_dir` as it runs. The
pipeline copies those files to the host before removing the container, including
files already written when an episode times out. Return `AgentRunResult` when
the adapter finishes. Its `success` field reports whether adapter execution
succeeded, not whether the model won the game.

## Register and install

Add an entry to the list in the games repository's `agent_registry.json`:

```json
{
  "agent_name": "example-with-my-model",
  "backend": "example",
  "agent_config": {"model": "my-model"}
}
```

`agent_name` is the name you pass to `agentclem --agent`. `backend` identifies
your adapter, and `agent_config` supplies its constructor arguments. Here,
`model` is passed directly to the example class. Built-in adapters also support
`clem_model`, which refers to a model entry in `model_registry.json` and uses
the credentials in `key.json`.

Install the harness software with an explicit version in
`src/clemagents/docker/agent-sandbox/Dockerfile`, then build from the pipeline repository:

```bash
docker build -f src/clemagents/docker/agent-sandbox/Dockerfile -t clemagents-sandbox:dev .
```

Python-only adapter edits take effect directly in an editable installation.
Rebuild when native software or dependencies change.

## Minimal trace

Return a dictionary with an ordered `events` list:

```json
{
  "events": [
    {"type": "reasoning", "content": "Recorded reasoning"},
    {"type": "tool_call", "name": "native_tool_name", "call_id": "1", "arguments": {"response": "guess"}},
    {"type": "tool_result", "name": "native_tool_name", "call_id": "1", "content": "Game feedback"}
  ]
}
```

Each event needs a non-empty `type` and JSON-serializable values. Preserve the
observed order, tool names and call IDs. Other supported types include
`assistant_text`, `instruction`, `error` and `trace_warning`. Unknown types remain
renderable. Use `agent_id` and `parent_call_id` for subagent events when available.

Return `missing_agent_trace(reason, backend)` from `clemagents.adapters.utils.parse`
when capture is unavailable, or add a `trace_warning` for partial capture.

Once you creted this function the engine takes care of the rest.
The inherited serializer validates events, fills in version/backend/sequence
fields, and writes `agent_loop.json`. `agentclem-transcribe -r test_results`
renders that JSON as HTML without loading the harness. Native parsing stays
inside the adapter; a new adapter needs no renderer changes.

For an initial implementation of a harness it is recommended to create a mock output for this function.
