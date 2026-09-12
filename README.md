# IMClemAgents

Evaluate agent harnesses on clembench games through a harness-agnostic pipeline
built on clemcore. The Python package is `clemagents`; the run command is `agentclem`.

Games, model configurations and experiment results live in a separate games
repository, such as [IMClemAgents-clembench](https://github.com/TimLeiber/IMClemAgents-clembench).

## Installation

**Docker is required** to build the sandbox and run harness episodes. Installing
the Python package does not install Docker. Git and Conda are also used below.
Complete the Docker setup for your OS before building the sandbox.

Clone the repository, then use Python 3.10–3.12 and a fresh environment:

```bash
git clone https://github.com/TimLeiber/IMClemAgents.git
cd IMClemAgents
conda create -n imagent python=3.11 pip -y
conda activate imagent
python -m pip install .
agentclem --help
```

The package depends on the published `clemcore` distribution. It does not need
the project's modified clemcore checkout. Dependency versions are initially
pinned where protocol compatibility matters; test upgrades before benchmarking.
Contributors can use `python -m pip install -e '.[dev]'` instead so Python edits
take effect without reinstalling. Ordinary installations need reinstalling after edits.

## Sandbox

Docker runs the harness and its auxiliary tools. macOS on Apple Silicon and
Ubuntu on AMD64 have been smoke-tested; other configurations are not claimed
as verified.

### macOS

Install [Docker Desktop for Mac](https://docs.docker.com/desktop/setup/install/mac-install/),
choosing the Apple Silicon or Intel download for your machine. Open Docker Desktop,
complete its setup and wait for the Docker engine to start. Keep it running while
building or playing episodes.

### Debian-based Linux

Install Docker Engine using the official instructions for
[Ubuntu](https://docs.docker.com/engine/install/ubuntu/) or
[Debian](https://docs.docker.com/engine/install/debian/). Follow the appropriate
distribution's repository setup and package-install commands; do not mix the two.
Docker Desktop is not required on Linux. After installation:

```bash
sudo systemctl enable --now docker
sudo groupadd --force docker
sudo usermod -aG docker "$USER"
```

Log out and back in, or reconnect your SSH session, before running `agentclem`.
The pipeline must be able to run Docker without sudo. Membership in the Docker
group grants root-level privileges; see
[Docker's Linux post-installation guide](https://docs.docker.com/engine/install/linux-postinstall/).
Ubuntu was tested; Debian instructions are provided but were not separately tested.

### Windows (untested)

Install [Docker Desktop for Windows](https://docs.docker.com/desktop/setup/install/windows-install/)
following its WSL 2 prerequisites. Enable the WSL 2 backend, Linux containers and
integration with your chosen WSL distribution. Run the repository, Conda setup
and commands below inside that WSL distribution, with Docker Desktop running.
The Windows/WSL setup has not been tested for this project.

### Verify Docker and build

Run from the terminal you will use for `agentclem` (inside WSL on Windows):

```bash
docker version
docker info
docker run --rm hello-world
```

`docker info` and `hello-world` must succeed without sudo. A CLI version alone
does not prove that the Docker engine is running or accessible.

Docker Compose is not required. The container runs Debian Bookworm Linux on every
host OS; its definition is `src/clemagents/docker/agent-sandbox/Dockerfile`.
Build the sandbox locally for your machine's architecture. No image-registry
account, image upload or multi-architecture build is required:

```bash
docker build -f src/clemagents/docker/agent-sandbox/Dockerfile -t clemagents-sandbox:dev .
```

The sandbox installs the supported harnesses. The runner mounts this package's
code read-only for each episode; changing Python source does not require a
rebuild for an editable installation, while changing installed tools or dependencies does.
This also works with a non-editable installation: the mounted code comes from the
installed package, not a required development checkout. Set `CLEMAGENTS_SANDBOX_IMAGE`
to select another locally built image; otherwise the runner uses
`clemagents-sandbox:dev`. Build it before running an experiment.

Harness versions are pinned in the Dockerfile: Codex CLI 0.152.0, Hermes 0.16.0,
and OpenClaw 2026.8.1. The Claude Code adapter uses `claude-agent-sdk` 0.2.87,
which bundles Claude Code 2.1.150; the separately installed standalone Claude Code
CLI is pinned to 2.1.252. The separate Codex Python SDK is pinned to 0.154.0.
Update these versions deliberately and repeat the integration checks before
using a rebuilt image for benchmarking. These pins do not lock every transitive
dependency or operating-system package.
The package and Hermes both require MCP `1.26.0`, the version used by the verified
sandbox smoke. Python dependencies are resolved together and `pip check` must pass
during the build; incompatible requirements stop the build rather than silently
replacing a previously installed dependency.

Adding a harness requires an adapter and installation of the harness software in
the Dockerfile, followed by a rebuild. No game or engine-specific dispatch changes
are needed. See [adding a harness](examples/add_harness/README.md) and the
[fresh-install verification steps](documentation/fresh-install.md).

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
