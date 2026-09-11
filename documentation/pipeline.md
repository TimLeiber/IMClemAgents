# Pipeline and dependency boundary

The host runs the game through clemcore. A Docker container runs the selected
harness. The harness connects to a small MCP bridge inside that container;
the bridge forwards game actions to the host and returns observations.

1. `agentclem` selects the game instances and agent registry entry.
2. The engine looks up the model registry entry; the adapter receives that entry
   and the agent settings through `resolve_model_connection(model_spec, agent_config)`.
3. The host starts the game server and creates a fresh container for each episode.
4. The harness calls `start_game`, then `submit_response` for subsequent actions.
5. Game observations return as text and, where supplied, image content. Image files
   are also materialized inside the container as observations arrive.
6. On completion or timeout, the pipeline finalizes the game and collects artifacts.
   Normal completion allows bounded artifact finalization. At timeout the container
   stops immediately; already-written artifacts are recovered without extending play.
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
| `adapters/openai_compatible_proxy.py` | Unmodified API payload recording/forwarding and shutdown gating |
| `transcribe_agent_loop/` | Harness-independent HTML rendering of uniform events |
| `docker/agent-sandbox/` | Sandbox image and container entry point |

Project-specific post-processing lives in the separate `IMClemAgents-clembench`
repository and is not included in this package.

The engine contains no game-name or harness-name dispatch. Native tool names,
model choices and request/response bodies are preserved. Harness source is not patched.

The game integration uses unmodified clemcore. It checks termination before
reading another turn and completes dead-player cleanup through the environment's
normal methods. It neither subclasses nor patches `AECToGymWrapper`.

## Configuration and traces

The registry's `agent_config` is passed to the adapter constructor. Native
options therefore belong to that adapter. See the constructor for its complete
option list and the games repository for registry examples.

The adapter's `resolve_model_connection` owns model and reasoning interpretation.
Shared resolution only looks up data, calls that method and checks its return type;
it does not apply a reasoning policy after the adapter returns. Direct model
configurations without `clem_model` skip this method.

Set `reasoning_effort` in the agent registry, not the model registry's vanilla
request settings. Each adapter uses the harness's native interface:

| Adapter | Native reasoning control | Native temperature control exposed here |
| --- | --- | --- |
| Codex | `model_reasoning_effort` in `config.toml` | Not supported |
| Claude Code | SDK `effort`; `none`/`off` requests native thinking disablement | Not supported |
| Hermes | `agent.reasoning_effort`; stops if an explicit control is omitted from the native request | Not supported by the installed CLI |
| OpenClaw | `--thinking` | `agents.defaults.models[model].params.temperature` |

Use values accepted by the installed harness and selected model. Omitted controls
leave native defaults intact; an effort label does not guarantee a token budget
or identical behavior across harnesses. See the native references for
[Codex](https://developers.openai.com/codex/config-reference/),
[Claude Code](https://code.claude.com/docs/en/agent-sdk/python),
[Hermes](https://hermes-agent.nousresearch.com/docs/user-guide/configuration/) and
[OpenClaw](https://docs.openclaw.ai/tools/thinking).

Explicit CLI settings are copied into an isolated run registry and passed to the
adapter constructor. They take precedence over registry settings without editing
the original registry. Unsupported constructor options fail before container
startup. Adding an adapter requires no central mapping of generation controls.

The recorder never injects `extra_body`, temperature, reasoning or a replacement
model. Stale connections requesting such overwrites are rejected. Built-in adapters
warn when a model registry contains vanilla generation settings; configure the
corresponding native agent controls instead. OpenClaw's custom model definition
may use registry capability metadata to select its native protocol, but does not
copy the vanilla thinking value into run parameters.

The real model ID is passed to the harness, including Claude Code's SDK. Subagent
model choices remain the harness's responsibility. The recorder still gates new
inference after game completion so artifact finalization cannot prolong play.
Hermes instead uses its native plugin observers and full request dumps, preserving
the original endpoint because its native reasoning support depends on that URL.
Its completion hooks and process-group shutdown stop further conversation work;
native session export collects the saved conversation afterwards. Capture is
explicitly partial: completed reasoning and tool events are available, but raw
SSE, interrupted streams, SDK-internal retries and auxiliary API calls can be absent.
Claude Code's native `off` omits `thinking` on the wire; compatible providers must
be checked for their interpretation of that omission.
A registry setting alone is not evidence of provider behavior: inspect recorded
requests when verifying reasoning effort, sampling settings or model routing.
Offline tests verify adapter configuration and byte-preserving transport, not
whether every provider honors every setting.
The [native request verification](offline-verification.md) records installed-harness
wire checks, integration corrections and the remaining live-provider checks.

`AgentRunResult.success` describes adapter execution. Game success, a legitimate
loss, a timeout and an early harness exit remain distinct in game/run metadata.

Trace flow: native artifacts → adapter `parse_agent_trace` → validated
`agent_loop.json` → `agentclem-transcribe` → `agent_loop.html`.
Native parsing stays in the adapter; the HTML command consumes JSON only.
The small shared contract lives in `adapters/traces/schema.py`. A parser need
only return an ordered `events` list; version, backend and sequence numbers can
be supplied by the serializer. Unknown event types remain renderable without
engine changes. See the [minimal trace example](../examples/add_harness/README.md#minimal-trace).

Missing capture, empty parser output and parser failures are visible warnings,
not silent empty transcripts. Invalid event structure is rejected at the boundary;
the pipeline preserves raw artifacts and writes a warning trace if parsing fails.
Optional capture metadata describes what was actually recorded, not what the
model necessarily generated. The contract does not enforce game outcomes or
change harness behavior.

The container publishes `artifacts_started.json` before adapter initialization,
identifying the current run's directory inside shared storage. After normal
finalization it publishes `artifacts_ready.json`. The host clears both markers
before each episode and validates their paths. If finalization never completes,
it still copies the announced directory before temporary storage is removed.
Such captures are labelled `partial` in run metadata and the HTML transcript;
files and the last event may be incomplete, and a session export may be absent.
Missing directories are labelled `unavailable`, never silently treated as complete.
This lifecycle is shared by every adapter; native file interpretation remains
in each adapter's parser.

## Code style

Keep short calls, definitions and collections inline. For longer expressions,
align continuation lines with the first argument or item after the opening
delimiter, and keep closing delimiters with the last item where practical.
Use 120 columns as a guideline, not a reason to split every argument onto its
own line. `.style.yapf` supplies formatter defaults; retain visual alignment
when reviewing nested collections. Write concise lowercase comments without
trailing punctuation. Keep docstrings in the existing `Args:` / `Returns:` style.

## Installation checks

The initial extraction is tested in an isolated virtual environment against
the PyPI `clemcore==3.7.2` wheel. No local clemcore checkout is installed or mounted.
The sandbox mounts only the extracted `clemagents` package and uses its own
installed clemcore dependency. A Docker build and live provider smoke test are
separate checks from the automated fixture tests.
