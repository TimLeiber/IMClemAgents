# IMClemAgents

Evaluate agent harnesses on clembench games through a harness-agnostic pipeline
built on clemcore. The Python package is `clemagents`; the run command is `agentclem`.

Games, model configurations and experiment results live in a separate games
repository, such as [IMClemAgents-clembench](https://github.com/TimLeiber/IMClemAgents-clembench).

## Installation

Use Python 3.10–3.12 and a fresh environment:

```bash
conda create -n imagent python=3.11 pip -y
conda activate imagent
python -m pip install -e '.[dev]'
agentclem --help
```

The package depends on the published `clemcore` distribution. It does not need
the project's modified clemcore checkout. Dependency versions are initially
pinned where protocol compatibility matters; test upgrades before benchmarking.

## Sandbox

Docker runs the harness and its auxiliary tools. From this repository:

```bash
docker build -f src/clemagents/docker/agent-sandbox/Dockerfile -t clemagents-sandbox:dev .
```

The sandbox installs the supported harnesses. The runner mounts this package's
code read-only for each episode; changing Python source does not require a
rebuild, while changing installed tools or dependencies does.

## Configuration and running

Run from a games repository containing `agent_registry.json` and
`model_registry.json`. `agent_config.clem_model` names a model in that registry.
Follow the games repository's instructions for agent configurations and
credentials. Keep its local `key.json` file ignored by Git.

```bash
agentclem \
  --game geolocate \
  --agent hermes-with-glm-5.3-flash \
  --instances_filename instances_geolocate_official \
  --experiment_name civic_public \
  --max-instances 1 \
  --results_dir test_results \
  --episode-timeout 1200
```

For a second, automated player, provide `--agent-player player_1 --models MODEL`.
`-l` controls the automated players' output limit; `--episode-timeout` is the
harness episode's wall-clock limit. A benchmark run calls the configured providers.

Reasoning and sampling controls use each harness's native interface, configured
in the agent registry. `--temperature` requires adapter support (currently OpenClaw);
it is not injected into API requests. See [native controls](documentation/pipeline.md#configuration-and-traces).

## Scoring and transcripts

Run in the games repository:

```bash
clem score -r test_results
clem eval -r test_results --std --sort clemscore
clem transcribe -r test_results
agentclem-transcribe -r test_results
```

## Development

```bash
python -m pytest -q
python -m build
```

Tests use fixtures and local scripted endpoints. Native harness tests are opt-in
and require the sandbox. See [documentation/pipeline.md](documentation/pipeline.md) for the execution flow and
`examples/add_harness/` for the adapter contract.
