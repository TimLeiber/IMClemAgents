# Pipeline and dependency boundary

The host runs the game through clemcore. A Docker container runs the selected
harness. The harness connects to a small MCP bridge inside that container;
the bridge forwards game actions to the host and returns observations.

1. `agentclem` selects the game instances and agent registry entry.
2. The adapter resolves its model connection using shared registry and credential helpers.
3. The host starts the game server and creates a fresh container for each episode.
4. The harness calls `start_game`, then `submit_response` for subsequent actions.
5. Game observations return as text and, where supplied, image content. Image files
   are also materialized inside the container as observations arrive.
6. On completion or timeout, the pipeline finalizes the game and collects artifacts.
   Container shutdown allows a bounded finalization period for native trace exports.
7. The adapter parses its artifacts into `agent_loop.json`. The shared renderer
   turns those events into HTML. clemcore writes the normal game interactions and scores.

## Files

| Location under `src/clemagents/` | Responsibility |
| --- | --- |
| `run_pipeline/` | CLI, episode scheduling, Docker lifecycle and artifact collection |
| `mcp/environment.py` | Game discovery and execution through clemcore's existing turn-based environment |
| `mcp/server.py` | Host server and clemcore result callbacks |
| `mcp/bridge.py` | Container-side MCP tools and observation delivery |
| `adapters/base.py` | Small common harness interface |
| `adapters/__init__.py` | Discover adapter class from the registry's backend name |
| `adapters/{backend}.py` | Native configuration, startup and artifact export for that harness |
| `adapters/traces/{backend}.py` | Parse that harness's native artifacts |
| `adapters/model_connection.py` | Shared model registry, credential and provider helpers |
| `adapters/utils.py` | Shared process and artifact utilities |
| `adapters/openai_compatible_proxy.py` | API recording, configured request overrides and shutdown gating |
| `transcribe_agent_loop/` | Harness-independent HTML rendering of uniform events |
| `docker/agent-sandbox/` | Sandbox image and container entry point |

Project-specific post-processing lives in the separate `IMClemAgents-clembench`
repository and is not included in this package.

The engine contains no game-name or harness-name dispatch. Native tool names
are preserved. API recording/overrides are existing transport behavior; this
extraction does not add tool-name translation or patches to harness source.

The game integration uses unmodified clemcore. It checks termination before
reading another turn and completes dead-player cleanup through the environment's
normal methods. It neither subclasses nor patches `AECToGymWrapper`.

## Configuration and traces

The registry's `agent_config` is passed to the adapter constructor. Native
options therefore belong to that adapter. See the constructor for its complete
option list; the registry template contains representative configurations.

The adapter's `resolve_model_connection` prepares native provider configuration.
Shared transport code retains the configured generation overrides. A registry
setting alone is not evidence of provider behavior: inspect recorded requests
when verifying reasoning effort, sampling settings or model routing.

`AgentRunResult.success` describes adapter execution. Game success, a legitimate
loss, a timeout and an early harness exit remain distinct in game/run metadata.
The transcript records missing capture as a warning rather than inventing events.

## Installation checks

The initial extraction is tested in an isolated virtual environment against
the PyPI `clemcore==3.7.2` wheel. No local clemcore checkout is installed or mounted.
The sandbox mounts only the extracted `clemagents` package and uses its own
installed clemcore dependency. A Docker build and live provider smoke test are
separate checks from the automated fixture tests.
