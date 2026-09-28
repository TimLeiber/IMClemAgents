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
The following image illustrated the dataflow when the model makes an MCP call to transition to the next game state.
It also includes information on where the docker, the MCP server The game environment with the other models live etc. 
[![Tool-call and result flow between the model, harness, MCP bridge and host game environment](images/architecture_tool_flow.png)](images/architecture_tool_flow.pdf)

-->
Tool-call and result flow for `submit_response` call, i.e. an MCP tool in the framework. Gray marks the host, blue the
Docker container running on it. The model endpoint may be on the host, on the
local network or remote. Click the diagram to open the PDF.

The harness is an MCP client. It starts the bridge as an MCP server over stdio.
The bridge forwards calls to the host's OpenEnv `/mcp` endpoint using HTTP.

OpenEnv assigns an identifier to each game run which in turn necessitated teh bridge on the docker container.
The bridge creates that run, adds its identifier to subsequent requests and closes it when the game ends.
This is needed even though the pipeline runs one episode at a time because of teh way the host endpoint expects input.

In steps 1–4, the model requests `submit_response`, the harness calls the bridge,
and the bridge forwards the response string to the host environment where a fucntion is called to act on the environment. The host
uses it as the player's next move in the clemcore game.

In steps 5–8, the resulting observation returns through the host endpoint and
bridge as MCP text and image content. The harness includes that tool result in
the next request to the model. Images are also written to
`/workspace/game_observations/` for tools that read local files. The harness
decides how this content enters the model's context.

The shared instruction is the `meta_prompt` field in
`adapters/utils/external_agent_config.yaml`. Game-specific prompts remain in
the game and are delivered through observations. This is where one would change the meta prompt.

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
