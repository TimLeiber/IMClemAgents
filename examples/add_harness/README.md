# Add a harness

Create `src/clemagents/adapters/example/`, add an empty `__init__.py`, and copy
[example.py](example.py) into it. Follow [base.py](../../src/clemagents/adapters/base.py)
and the built-in harness directories:

```text
example/
    __init__.py
    example.py     # one ExternalAgentHarness subclass
    utils.py       # optional native configuration and execution helpers
    parse.py       # optional native artifact parser
```

The registry's `backend` selects `adapters/<backend>/<backend>.py` automatically.
Adding a harness requires no changes to discovery, shared utilities or the engine.

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

Configure the harness to launch `python -m clemagents.mcp.bridge` using the
environment from `mcp_environment`. Use the shared process/completion helpers
to stop when the game ends and preserve partial artifacts on timeout. Store
artifacts under `output_dir`; the container entry point publishes the final
artifacts-ready marker. `AgentRunResult.success` describes adapter execution,
not whether the model won the game.

## Register and install

Add an entry to the games repository's `agent_registry.json`:

```json
{
  "agent_name": "example-with-my-model",
  "backend": "example",
  "agent_config": {"model": "my-model"}
}
```

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
`assistant_text`, `instruction`, `error` and `trace_warning`; unknown types remain
renderable. Use `agent_id` and `parent_call_id` for subagent events when available.

Return `missing_agent_trace(reason, backend)` from `clemagents.adapters.utils.parse`
when capture is unavailable, or add a `trace_warning` for partial capture.
Only include reasoning and other output actually recorded by the harness.

The inherited serializer validates events, fills in version/backend/sequence
fields, and writes `agent_loop.json`. `agentclem-transcribe -r test_results`
renders that JSON as HTML without loading the harness. Native parsing stays
inside the adapter; a new adapter needs no renderer changes.

Check the parser against saved native output, then run one game episode into
`test_results` and inspect its tool calls, artifacts and transcript. A game win
is not required to verify the integration.
