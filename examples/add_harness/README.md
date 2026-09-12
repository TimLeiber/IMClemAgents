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
Install the native harness software in
`src/clemagents/docker/agent-sandbox/Dockerfile`, pinning its version explicitly.
Resolve any shared Python dependency constraints together with the existing
packages; do not bypass conflicting requirements with `--no-deps`. Rebuild locally:

```bash
docker build -f src/clemagents/docker/agent-sandbox/Dockerfile -t clemagents-sandbox:dev .
```

Adding an adapter alone does not install the harness software. The Dockerfile
change is sandbox setup, not an engine or game change. Rebuild whenever the
harness version or its dependencies change; a Python-only adapter edit in an
editable installation does not require rebuilding.

The adapter has three responsibilities:

- `run_episode(instruction, output_dir)` configures MCP through the harness's
  documented interface, runs the harness, and returns `AgentRunResult` with saved artifacts.
- `parse_agent_trace(episode_dir, metadata)` converts native logs into a dictionary
  containing an ordered `events` list. The common serializer fills in the version,
  adapter name and event sequence numbers when omitted.
- `resolve_model_connection(model_spec, agent_config)` is required only when
  the configuration uses `clem_model`. The engine looks up the model and passes
  its registry entry and all agent settings as independent dictionaries. Interpret
  those settings inside the adapter and prepare its native model connection;
  shared resolution returns your dictionary unchanged. Credential helpers are
  available, but model selection and reasoning semantics belong to the adapter.

The connection is serialized to the container and read through
`load_model_connection`. Existing adapters use fields such as `model`,
`base_url` and `env`; add native fields as needed. Native configuration should
use the harness's documented settings, not provider-request rewriting. The
optional recorder forwards request and response bodies unchanged and rejects
legacy overwrite fields. Expose supported controls as constructor arguments;
the runner passes them through without knowing their meaning. An explicit CLI
temperature is passed as `temperature` and rejected if your constructor does
not support it. Do not accept a control that your native interface cannot apply.

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
  "events": [
    {"type": "reasoning", "content": "Recorded reasoning"},
    {"type": "tool_call", "name": "native_tool_name", "call_id": "1", "arguments": {"response": "guess"}},
    {"type": "tool_result", "name": "native_tool_name", "call_id": "1", "content": "Game feedback"}
  ]
}
```

That is sufficient: no schema classes or builder framework are required.
Each event needs a non-empty `type`; text and structured output go in `content`,
tool inputs in `arguments`. Values must be JSON-serializable. Preserve observed
event order, actual tool names and call IDs; never invent missing output or
reorder calls to make the run appear successful.

Other useful types are `assistant_text`, `instruction`, `error` and `trace_warning`.
A failed tool result is still a `tool_result`: add `status: "failed"` or
`is_error: true` and keep the actual error in `content`. Record each observed
retry separately. `agent_id` and `parent_call_id` optionally identify subagent
events; they do not introduce a second event format. Existing `agent` labels
are also supported.

Unknown event types and extra fields are preserved and rendered generically.
Optional `model_request`/`model_response` events go in the raw API appendix
(`raw` contains the recorded body); `tool_definitions` uses a `tools` list.
The renderer hides embedded media bytes and bounds raw API previews for readability;
the JSON and native artifacts retain the original captured data.

For missing capture, return `missing_agent_trace(reason, backend)` from
`clemagents.adapters.traces.schema`, or include a `trace_warning` event for a
partial capture. Optional `capture` metadata can describe individual sources,
for example `{"model_responses": {"status": "unavailable"}}`. Missing reasoning
capture is not evidence that the model did not reason.

The inherited serializer validates the contract and writes `agent_loop.json`.
Malformed output or parser exceptions produce a visible warning trace during
pipeline persistence; raw artifacts remain available. `agentclem-transcribe -r test_results`
reads only the saved JSON and renders HTML, without loading a
harness or looking up the registry. No renderer changes are needed for a new adapter.

First test with native-output fixtures and scripted local responses. Then run
one real game episode into `test_results` and inspect both its game result and
agent-loop transcript before benchmarking a matrix.
