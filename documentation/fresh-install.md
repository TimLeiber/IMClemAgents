# Verify a fresh installation

After committing and pushing the intended revision, repeat this on Ubuntu and
macOS. Docker must be installed, running and usable without sudo. Windows with
Docker Desktop's WSL 2 backend and Linux containers is not yet verified.

Use a new terminal with no sandbox-image override. Keep the existing working
environments and official results unchanged. Run from a parent directory where
`clemagents-verification` does not already exist. Use any supported environment
manager; the example uses Python's built-in `venv`, not a project-specific environment:

```bash
mkdir clemagents-verification
cd clemagents-verification
git clone https://github.com/TimLeiber/IMClemAgents.git
cd IMClemAgents
git checkout COMMIT_SHA
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install .
python -m pip check
python -c 'import clemagents; print(clemagents.__file__)'
agentclem --help
docker build -f src/clemagents/docker/agent-sandbox/Dockerfile -t clemagents-sandbox:verify .
docker run --rm --pull never --network none clemagents-sandbox:verify python -m pip check
export CLEMAGENTS_SANDBOX_IMAGE=clemagents-sandbox:verify
```

Replace `COMMIT_SHA` with the committed revision. The package path must be inside this new
environment, not the checkout. Docker builds the host architecture; cached layers
are fine and do not reuse the old host Python environment. The separate image tag
preserves the existing development image.

From this new pipeline checkout, prepare a separate games checkout:

```bash
cd ..
git clone https://github.com/TimLeiber/IMClemAgents-clembench.git
cd IMClemAgents-clembench
```

Configure a local ignored `key.json` following that repository's instructions.
Never commit credentials. Use the existing GLM/Codex agent and one official
instance; this does not require a private smoke-instance file:

```bash
agentclem --game geolocate --agent codex-with-glm-5.3-flash \
  --instances_filename instances_geolocate_official --experiment_name civic_public \
  --max-instances 1 --results_dir test_results --episode-timeout 300
agentclem-transcribe -r test_results
```

Confirm the CLI selects one episode. Check a valid game submission, normal game
termination, finalized artifacts and a readable transcript containing recorded
requests, responses and tool activity. A win is not required; a timeout does not
verify normal termination. Record the commit, image ID, OS and architecture.

## Cleanup after both checks

Only after inspecting both runs, deactivate the temporary Python
environment, unset `CLEMAGENTS_SANDBOX_IMAGE`, and remove the `clemagents-sandbox:verify`
image tag and the specifically created `clemagents-verification` directory (including
its `.venv` environment). Resolve
the exact paths and environment names before deleting anything. Do not remove the
original repositories, working environments or official results. Copy any evidence
you want to retain before removing the temporary results and credentials.
