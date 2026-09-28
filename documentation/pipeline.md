# Pipeline

The framework connects clemcore games to external agent harnesses. The game
runs on the host. Each episode gets a new Docker container containing the
harness, its adapter, its tools and the MCP bridge. Model inference runs at the
configured endpoint, which may be local or remote.

## Running an episode

1. `agentclem` loads the selected game instances and agent registry entry.
2. If `clem_model` is set, the engine looks up the model and credentials.
   The adapter converts them into the harness's native configuration.
3. The pipeline starts the host game server and runs the instances sequentially,
   creating a fresh container for each one.
4. Inside the container, the adapter launches the harness with its native settings,
   the shared meta prompt and a connection to the MCP bridge.
5. The harness calls `start_game` once to receive the initial observation.
   It then calls `submit_response(response)` for each game move.
6. When the game ends or the episode times out, the pipeline collects the
   available artifacts and removes the container.

Game rules, state and other players are handled by clemcore on the host.
Adding a game therefore uses the normal Clembench game structure, not a harness
adapter. Games, registries and experiment analysis live in the separate
`IMClemAgents-clembench` repository.

## MCP interface

The harness is an MCP client. It starts the bridge as an MCP server over stdio.
The bridge forwards calls to the host's OpenEnv `/mcp` endpoint using HTTP
JSON-RPC.

OpenEnv assigns an identifier to each game run. The bridge creates that run,
adds its identifier to subsequent requests and closes it when the game ends.
This is needed even though the pipeline runs one episode at a time.

A `submit_response` call passes the response string to the host environment,
which advances the clemcore game. The resulting observation travels back through
the bridge as MCP text and image content. Images are also written to
`/workspace/game_observations/` for tools that read local files. The harness
decides how this content enters the model's context.

The shared instruction is the `meta_prompt` field in
`adapters/utils/external_agent_config.yaml`. Game-specific prompts remain in
the game and are delivered through observations.

## Package structure

Paths below are relative to `src/clemagents/`.

| Directory | Responsibility |
| --- | --- |
| `run_pipeline/` | CLI, episode scheduling, Docker lifecycle and artifact collection |
| `mcp/` | Host game environment, server and container-side MCP bridge |
| `adapters/<backend>/` | Harness class, native configuration and artifact parsing |
| `adapters/utils/` | Shared model lookup, connection, process and trace helpers |
| `transcribe_agent_loop/` | HTML rendering of parsed agent traces |
| `docker/agent-sandbox/` | Dockerfile and container entry point |

Each adapter has its own directory with `<backend>.py`, `__init__.py` and
optional `utils.py` and `parse.py` files. The registry's `backend` selects the
class automatically. Adding a harness does not require changing shared dispatch
code. See [adding a harness](../examples/add_harness/README.md).

## Configuration and traces

The registry's `agent_config` supplies constructor arguments to the adapter.
Explicit CLI settings override them for that run without editing the registry.
The constructor lists the available options.

With `clem_model`, shared code looks up the model entry and credentials, then
calls the adapter's `resolve_model_connection` method. The adapter owns native
model and generation settings. Direct model configurations skip this lookup.

Set reasoning and sampling controls in `agent_config`, not in the model
registry's vanilla generation settings. Omitted controls retain native defaults.

| Adapter | Reasoning control | Temperature control |
| --- | --- | --- |
| Codex | Native `model_reasoning_effort` | Not exposed |
| Claude Code | SDK `effort`, with `none`/`off` requesting thinking disablement | Not exposed |
| Hermes | Native `agent.reasoning_effort` | Not exposed by the installed CLI |
| OpenClaw | Native `--thinking` | Native model `params.temperature` |

Supported values depend on the harness and provider. The same reasoning label
need not produce the same behavior across harnesses. Hermes reports unsupported
explicit reasoning settings rather than silently dropping them. Recorded
requests show which controls were sent, not whether the provider honored them.

### TLS

For additional trusted certificates, set `agent_config.ca_bundle` to a public
PEM file on the host. Its certificates are added to the container's trust store.

Alternatively, `agent_config.verify_tls: false` disables certificate verification
for that agent's model connection. Use this only for a trusted endpoint when
necessary. Both options require `clem_model` and cannot be combined.
Credentials remain in `key.json`.

Hermes normally connects directly and uses forwarding when verification is
disabled. These connection settings do not change model request bodies.

### Codex context

Registry `context_size` is forwarded to Codex's native `model_context_window`.
An explicit `agent_config.model_context_window` overrides it.
`model_auto_compact_token_limit` optionally sets the native compaction threshold.
Both values are integer token counts. They configure Codex, not the inference
server's context allocation.

Unknown models use Codex's native fallback. The adapter does not generate model
catalogs. An optional `agent_config.model_catalog` supplies a complete native
JSON catalog containing the selected model's `slug`.

## Results and artifacts

The host saves game instances and interactions through clemcore callbacks.
Harness artifacts are collected separately into the episode's result directory.
Scoring remains a separate Clembench step.

The container announces its artifact directory before adapter initialization
and marks it ready after finalization. Normal completion allows time to finish
writing artifacts. At timeout, the pipeline stops the container and recovers
files already written. Capture may therefore be partial or unavailable.
This status is recorded in metadata and shown in the transcript.

Each adapter's `parse_agent_trace` method turns native artifacts into ordered
events. The shared serializer validates them and writes `agent_loop.json`.
`agentclem-transcribe` renders this file as `agent_loop.html`, without running
the model or changing the recorded game.

Capture depends on the harness. Hermes uses native observers and request dumps,
which may omit interrupted streams, internal retries or auxiliary API calls.
Missing trace content does not establish that an action did not occur.
Parsing failures produce visible warnings while preserving raw artifacts.

`AgentRunResult.success` describes adapter execution, not game success.
Wins, losses, timeouts and premature harness exits are recorded separately.
