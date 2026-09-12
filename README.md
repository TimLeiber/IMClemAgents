# IMClemAgents

Evaluate agent harnesses on clembench games through a harness-agnostic pipeline
built on clemcore. The run command is `agentclem`.

Games, model configurations and experiment results live in a separate games
repository, such as [IMClemAgents-clembench](https://github.com/TimLeiber/IMClemAgents-clembench).

## Installation

Requirements: **Python 3.10–3.12, pip, Git and Docker**. Installing this package
does not install Docker; see the OS-specific setup below. No particular Python
environment manager is required.

Clone the repository:

```bash
git clone https://github.com/TimLeiber/IMClemAgents.git
cd IMClemAgents
```

Use a Python environment of your choice. An isolated environment is recommended;
for example, with Python's built-in `venv` on macOS, Linux or WSL:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Install the package with your selected Python interpreter:

```bash
python -m pip install .
agentclem --help
```

The installation includes `clemcore` as a dependency; no separate clemcore
checkout is needed. For an editable contributor installation, see Development.

## Sandbox

Instances are run in a sanboxed environment so that an agent cannot corrupt the host system. We use Docker to run the harness and its auxiliary tools.
MacOS on Apple Silicon and
Ubuntu on AMD64 have been tested. Other configurations are not verified in the same way.

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

Log out and back in to apply the group change. This allows `agentclem` to run
Docker without sudo. The Docker group grants administrator-level access;
see [Docker's post-installation guide](https://docs.docker.com/engine/install/linux-postinstall/).

### Windows (untested)

Install [Docker Desktop for Windows](https://docs.docker.com/desktop/setup/install/windows-install/)
following its WSL 2 prerequisites. Enable the WSL 2 backend, Linux containers and
integration with your chosen WSL distribution. Run the repository, Python setup
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

The container runs Debian Bookworm Linux on every
host OS. Its definition is `src/clemagents/docker/agent-sandbox/Dockerfile`.
Build the sandbox locally for your machine's architecture.

```bash
docker build -f src/clemagents/docker/agent-sandbox/Dockerfile -t clemagents-sandbox:dev .
```
*If you add a new, i.e. previously unsupported harness you would also have to add it to the defintion and start a new build*.
Build once before running experiments.

The runner uses `clemagents-sandbox:dev` by default. To use another locally built image set
`CLEMAGENTS_SANDBOX_IMAGE`.

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
  --episode-timeout 600
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

From the pipeline repository, install development dependencies in editable mode:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m build
```

The runner mounts the installed Python package read-only into each container.
Editable Python changes therefore take effect without rebuilding the image;
non-editable installations need reinstalling after source changes.

Adding a harness requires an adapter and installation of its native software in
the Dockerfile, followed by a rebuild. No game or engine-specific dispatch changes
are needed. See [adding a harness](examples/add_harness/README.md).

Tests use fixtures and local scripted endpoints. Native harness tests are opt-in
and require the sandbox. See [pipeline documentation](documentation/pipeline.md)
for the execution flow and [fresh-install checks](documentation/fresh-install.md)
for release verification.
