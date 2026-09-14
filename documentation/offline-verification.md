# Native request verification

Checked on 2026-09-11 with Codex CLI 0.152.0, Claude Code SDK subprocess 2.1.150,
Hermes 0.16.0 and OpenClaw 2026.8.1. These findings are version-specific.
The existing `clemagents-sandbox:dev` image was used with updated package source
mounted read-only; no native harness source was changed.
Verification: 210 offline tests passed. The earlier full native fixture run
passed 15 checks; all eight current Hermes native fixtures also passed,
including truncated snapshots, unsupported GLM controls and provider defaults.
Timeout regressions use local subprocess fixtures, not Docker or model inference.
They cover interrupted artifact recovery, stale and unsafe paths, normal completion,
and rendering recovered native events with an explicit partial-capture warning.
The separate pre-existing native-startup test was not enabled in this run.

## Method

Each native harness ran in a fresh container with `--network none`, dummy keys
and scripted HTTP responses on loopback. OpenClaw also performs an independent
HTTPS OpenRouter capability lookup; that hostname was mapped to a local fixture
with a temporary test certificate. No real provider, model server, inference or
benchmark episode was used. Scripted terminal assistant text is not a successful
game-playing smoke test.

For Codex, Claude Code and OpenClaw, recorder ingress and mock-provider request
bodies were byte-identical, including model IDs, tools and generation fields.
Hermes uses native observation hooks and request dumps instead: the full agent
request dumps matched the fixture's received model, messages, tools and reasoning
fields after the SDK's normal `extra_body` merge. No adapter merged these fields
into a live request.

## Actual outgoing controls

Codex context verification: registry context limits now reach native `config.toml`
through its adapter. Both compatible and OpenRouter scripted-endpoint checks accepted
`model_context_window = 65536`, retained native reasoning controls and game definitions,
and preserved byte-identical recorder ingress/upstream request bodies. Focused unit
checks cover registry parsing, explicit overrides, validation and pre-inference artifacts.
The native `debug models` command verifies the installed catalog format without inference.
Automatic catalog generation and the copied native fallback profile have been removed.
Only explicitly supplied catalogs are forwarded; unknown models retain Codex's native
fallback and warning. Regression checks ensure the adapter neither queries a CLI version
nor generates a catalog during startup, while preserving context and compaction controls.
These checks do not demonstrate actual long-context compaction and do not alter past
benchmark results.

| Harness / route | Observed request | Outcome |
| --- | --- | --- |
| Codex / compatible and OpenRouter | `reasoning.effort: low`; also `none` locally and `high` on OpenRouter | forwarded |
| Claude Code / compatible and OpenRouter | `thinking.type: adaptive`, `output_config.effort: low`; OpenRouter `max` also checked | forwarded |
| Claude Code / compatible, `off` | no `thinking` field; native `output_config.effort: high` remains | provider interpretation still requires verification |
| Hermes / OpenRouter Qwen | `reasoning: {enabled: true, effort: low}` | forwarded with the original provider URL |
| Hermes / OpenRouter GLM, explicit `high` | native model-family allowlist excludes GLM and omits reasoning control | unsupported-control error before inference |
| Hermes / OpenRouter GLM, effort omitted | no reasoning override | native provider defaults, not an explicitly verified effort |
| Hermes / explicit `max` | native effort parser rejects this value | setup error before chat |
| Hermes / compatible Gemma, explicit `low` | native Hermes omits reasoning control | adapter stops before inference with an unsupported-control error |
| Hermes / compatible Gemma, effort omitted | native provider defaults | request and completed reasoning captured |
| OpenClaw / compatible Gemma | `chat_template_kwargs.enable_thinking: true` for `low`, `false` for `off`; temperature `0.7` | forwarded through native configuration |
| OpenClaw / OpenRouter | `reasoning.effort: low`, temperature `0.7` | forwarded |

Fixtures use `gemma4:26b-64k`, `qwen/qwen3.8-27b` and `z-ai/glm-5.3-flash` as identifiers only.
Gemma's boolean thinking flag is not a graduated low effort. A forwarded field
does not prove that Ollama or another compatible provider honors it.

Native game names were preserved: Codex's `mcp__game` namespace, Claude Code's
`mcp__game__start_game`, Hermes's `mcp_game_start_game`, and OpenClaw's
`game__start_game`, with the corresponding `submit_response` definitions.

## Integration corrections

- **Hermes registration:** configure native `mcp_servers` YAML and verify the
  actual tool inventory before chat. A failed CLI registration cannot silently
  produce a run without the game tools.
- **Hermes observation:** retain the real endpoint because Hermes uses it when
  deciding reasoning support. Native plugin observers replace the URL-changing
  recorder. Native hooks guard subsequent conversation API/tool work after the
  completion marker; the adapter also terminates the process group and exports
  the saved session. No prompt, response or tool-name transformations are used.
- **OpenClaw startup:** set the selected model as the profile's native default
  before Gateway startup, not merely an individual call override. Otherwise the
  Gateway can initialize with a different default and reject the requested
  thinking level. Explicitly disable automatic installation of Perplexity: this
  adapter already selects the installed DuckDuckGo search provider.
- **Claude Code `off`:** retain native SDK disablement and warn for compatible
  providers. Do not inject a different provider payload to simulate support.

## Capture and protocol limits

Hermes records completed normalized assistant responses, exposed internal
reasoning, tool calls/results, conversation retry events and subagent identities.
Full native request dumps are retained alongside its observer log. The parser
still emits the same harness-independent event contract as other adapters.

These observations are **not full HTTP/SSE capture**. Interrupted response streams,
SDK-internal retries and native auxiliary calls outside the conversation hooks
(including session-title generation) can be missing. Observer snapshots can
truncate schemas; full native request dumps retain the pre-SDK schemas separately.
Capture is labelled partial and unfinished requests generate visible warnings.
Incomplete observer snapshots do not prove that a reasoning control is absent.
They produce an explicit unverified-snapshot warning without blocking inference;
the full native request dump remains the source for subsequent verification.
The `high-large` fixture forces native snapshot truncation with an oversized Qwen
request and a reduced native snapshot limit; unit checks also cover the original
50,000-character preview shape. Truncated previews must not abort an episode.
The installed Hermes parser does not accept `max`; setup rejects it explicitly
instead of allowing the native fallback. Registry levels are not silently remapped.
Its `_supports_reasoning_extra_body` method also excludes `z-ai/` from the
OpenRouter model-family allowlist. Consequently, choosing `high` does not solve
GLM effort forwarding. Omitting the override allows native provider defaults;
it must not be reported as an explicitly forwarded `max` or `high` setting.
Hermes rejects stale request-body overrides. The later opt-in `agent_config.verify_tls`
control uses adapter-side forwarding when false; ordinary verified connections
retain native direct transport. See the TLS follow-up below.

## TLS follow-up

The university endpoint's incomplete certificate chain failed verification in both
Mac Python and sandbox Python. Installed vanilla clemcore uses `verify=False` for
its compatible backend, explaining why it connected. `agent_config.verify_tls: false`
now selects the same upstream policy without adding fields to `key.json`.

Metadata-only checks against the university server succeeded through the sandbox's
native Hermes client transport with this setting. No model completion was requested.
Paired network-disabled HTTPS fixtures compare native Hermes against a trusted
certificate route and a verification-disabled forwarding route, using
`Qwen/Qwen3.8-27B-FP8`. Only per-session home paths are normalized when comparing
tool definitions; outgoing request bodies are checked byte-for-byte through forwarding.
These checks are specific to the tested model/provider route and do not establish
equivalence for every endpoint-dependent native capability policy.

Claude Code requires Anthropic Messages; Codex requires Responses;
Hermes/OpenClaw use Chat Completions. A server advertising OpenAI compatibility
may support only the last protocol. The recorder does not translate them.
See [Ollama's Anthropic compatibility documentation](https://docs.ollama.com/api/anthropic-compatibility).

Temperature is exposed only by OpenClaw in these adapters. Other adapters reject
explicit temperature; unexposed `top_p` and `max_tokens` are also rejected instead
of injected. Native output limits observed here are harness defaults.
`max_turns`, permissions, deadlines and capture flags are execution controls,
not necessarily model request fields.

Routing unit checks cover loopback, IPv6 loopback, a tailnet address and remote
HTTPS. Only host loopback becomes `host.docker.internal`; this is not proof of
connectivity, authentication or TLS behavior on a real server.

## TODO: native Hermes reasoning effort

Compare the installed sandbox CLI's version and native GLM request construction
with Hermes Desktop. The tested CLI rejects `max` and omits GLM reasoning controls
through its model-family allowlist. The current Hermes–GLM smoke uses provider
defaults, not an explicitly forwarded effort. Verify a supported native route or
upstream version before restoring an override; do not inject settings through a
proxy or patch native harness source. This is separate from artifact recovery
and transcript parsing.

## Repeat and then smoke-test

```bash
python -m pytest -q
RUN_NATIVE_WIRE_TEST=1 python -m pytest -q tests/test_native_wire_requests.py
```

The opt-in native suite uses the existing image without building or pulling.
Each container has a 100-second bound. Reports go to pytest's temporary directory,
not official results. Use `-k` to select a harness or route.
The `completed` fixture writes a scripted terminal marker during the first
Hermes response and verifies that no subsequent tool or inference request runs.
It tests shutdown mechanics, not gameplay.

The runner mounts the current package source into the sandbox, including the
Hermes observer plugin. These Python-only corrections do not require rebuilding
an image that already contains the tested native harness versions and dependencies.
Then run separate, user-launched single-episode tests against a hosted API and
the personal model server. Check protocol acceptance, image/tool exchange,
native request controls, response reasoning and terminal artifact collection.
Neither an effort label nor a short response alone proves provider compliance.
