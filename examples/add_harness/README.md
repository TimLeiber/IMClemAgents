# Add a harness

Copy `example.py` to `src/clemagents/adapters/example.py`. Replace the marked
methods with your harness's native configuration, invocation and trace parser.
Keep exactly one `ExternalAgentHarness` subclass in that module.

Add an entry to the **games repository's** `agent_registry.json`:

```json
{
  "agent_name": "example-with-my-model",
  "backend": "example",
  "agent_config": {"model": "my-model"}
}
```

The backend determines the adapter module. Every `agent_config` field is passed
to its constructor. No runner or transcription dispatch table needs changing.
Install the native harness in the sandbox image as required; that dependency
installation may require a Dockerfile change and rebuild.

The adapter has three responsibilities:

- `run_episode(instruction, output_dir)` configures MCP through the harness's
  documented interface, runs the harness, and returns `AgentRunResult` with saved artifacts.
- `parse_agent_trace(episode_dir, metadata)` converts native logs into a dictionary
  with `schema_version`, `backend` and an ordered `events` list.
- `resolve_model_connection(clem_model)` is required only if the configuration
  uses a clemcore registry model via `clem_model`. Prepare the endpoint and
  environment your harness expects, using the shared provider helpers where useful.

Use `python -m clemagents.mcp.bridge` as the MCP server command, with the
environment returned by `mcp_environment`. Preserve the harness's native tool
definitions, prompts and model behavior. Keep any native log parser next to the
adapter, optionally under `adapters/traces/`.

Use the shared completion/process helpers to stop after the game ends and
preserve partial output on timeout. Keep artifacts under `output_dir` and
return their paths. The container entry point owns the final artifacts-ready marker.

## Minimal trace

```json
{
  "schema_version": 1,
  "backend": "example",
  "events": [
    {"sequence": 1, "type": "reasoning", "content": "Recorded reasoning"},
    {"sequence": 2, "type": "tool_call", "name": "native_tool_name", "call_id": "1", "content": {"response": "guess"}},
    {"sequence": 3, "type": "tool_result", "name": "native_tool_name", "call_id": "1", "content": "Game feedback"}
  ]
}
```

Other useful event types are `assistant_text`, `instruction`, `error`,
`trace_warning`, `model_request` and `model_response`. Keep actual tool names
and call IDs. Add `agent_id` to identify subagent events when the harness records
them. Report incomplete capture; never reconstruct unobserved model output.

First test with native-output fixtures and scripted local responses. Then run
one real game episode into `test_results` and inspect both its game result and
agent-loop transcript before benchmarking a matrix.
