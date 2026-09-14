import json
import inspect
import queue
import shutil
import socket
import subprocess
import threading
import time
import os
import sys
import traceback
from datetime import datetime, timezone
from multiprocessing import Process, Queue as ProcessQueue
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit  # needed when model provider is host, e.g. model use through ollama

from clemcore.clemgame.instances import GameInstances
from clemcore.clemgame.registry import GameRegistry

from clemagents.mcp.server import run_clem_mcp_server
from clemagents.adapters import harness_class_for_agent
from clemagents.adapters.utils.model_connection import resolve_agent_model_connection
from clemagents.adapters.utils.parse import missing_agent_trace, normalize_agent_trace
from clemagents.mcp.bridge import OpenEnvMCPClient

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SANDBOX_DIR = PACKAGE_ROOT / "docker" / "agent-sandbox"
# paths required by the external-agent pipeline
REGISTRY_PATH = Path("agent_registry.json").resolve()
KEYS_PATH = Path("key.json").resolve()
DOCKER_IMAGE = os.environ.get("CLEMAGENTS_SANDBOX_IMAGE", "clemagents-sandbox:dev")
# by convention i set 8001 to be port of the mcp server
SERVER_PORT = 8001
# define variables dependent on port
OPENENV_MCP_URL = f"http://host.docker.internal:{SERVER_PORT}/mcp"
HOST_OPENENV_MCP_URL = f"http://127.0.0.1:{SERVER_PORT}/mcp"
# debug message
AGENT_TERMINATION_REASON = ("AGENT_CONTROL_ERROR: external harness ended before completing the game")
EPISODE_TIMEOUT_REASON = ("AGENT_EPISODE_TIMEOUT: host wall-clock limit reached")
ARTIFACT_FINALIZATION_TIMEOUT = 90.0
ARTIFACT_EXIT_GRACE = 2.0


def _run_mcp_server_process(server_kwargs: dict, startup_status_queue) -> None:
    """Run the MCP server and report a child-process traceback on startup failure."""

    try:
        run_clem_mcp_server(startup_status_queue=startup_status_queue, **server_kwargs)
    except BaseException as error:
        try:
            startup_status_queue.put_nowait({"state": "failed",
                                             "error": f"{type(error).__name__}: {error}",
                                             "traceback": traceback.format_exc()})
        except Exception:
            pass
        raise


def _drain_server_startup_status(startup_status_queue, statuses: list[dict]) -> None:
    """Collect all currently available MCP-server startup reports."""

    while True:
        try:
            status = startup_status_queue.get_nowait()
        except queue.Empty:
            return
        except (EOFError, OSError, ValueError):
            return

        if isinstance(status, dict):
            statuses.append(status)


def _format_server_startup_status(statuses: list[dict]) -> str:
    """Format the most recent MCP-server startup report for an exception."""

    if not statuses:
        return "No startup status was received from the MCP-server process."

    return "Last MCP-server startup status: " + json.dumps(statuses[-1], ensure_ascii=False)


def _stop_docker_container(process: subprocess.Popen, cidfile_path: Path | None) -> None:
    """Stop one episode container and its complete process tree."""
    container_id = ""

    if cidfile_path is not None:
        deadline = time.monotonic() + 2

        while time.monotonic() < deadline and not cidfile_path.exists():
            time.sleep(0.05)

        if cidfile_path.exists():
            container_id = cidfile_path.read_text(encoding="utf-8").strip()

    try:
        if container_id:
            try:
                subprocess.run(["docker", "stop", "-t", "1", container_id], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, check=False, timeout=10)
            except (subprocess.TimeoutExpired, OSError):
                try:
                    subprocess.run(["docker", "kill", container_id], stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, check=False, timeout=10)
                except (subprocess.TimeoutExpired, OSError):
                    # docker desktop can temporarily stop answering control
                    # commands. the attached client process must still be
                    # bounded so one episode cannot terminate the matrix
                    pass
    finally:
        if process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except (OSError, ProcessLookupError):
                    pass
            except (OSError, ProcessLookupError):
                pass


def _read_json_marker(path: Path | None) -> dict | None:
    """Read one atomically published episode marker when available."""

    if path is None or not path.exists():
        return None

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    return value if isinstance(value, dict) else None


def _shared_artifact_directory(shared_state_dir: Path | None, marker: dict | None) -> Path | None:
    """Resolve a marker's relative artifact directory without escaping its root."""

    if shared_state_dir is None or not isinstance(marker, dict):
        return None

    relative_value = marker.get("artifact_directory")

    if not isinstance(relative_value, str) or not relative_value:
        return None

    relative_path = Path(relative_value)

    if relative_path.is_absolute() or ".." in relative_path.parts:
        return None

    candidate = (shared_state_dir / relative_path).resolve()
    shared_root = shared_state_dir.resolve()

    if shared_root not in candidate.parents:
        return None

    return candidate if candidate.is_dir() else None


def _abort_unfinished_episode(shared_state_dir: Path | None, experiment_name: str, game_id: int | str,
                              trace_lines: list[str], reason: str) -> dict | None:
    """Finalize an unfinished game through the host MCP server."""
    if shared_state_dir is None:
        return None

    session_path = shared_state_dir / "openenv_session.json"
    game_started_path = shared_state_dir / "game_started.json"
    client = OpenEnvMCPClient(HOST_OPENENV_MCP_URL)

    try:
        if session_path.exists():
            session_data = json.loads(session_path.read_text(encoding="utf-8"))
            session_id = session_data.get("session_id")

            if not session_id:
                raise ValueError(f"Session marker has no session_id: {session_path}")

            client.session_id = session_id
        elif not game_started_path.exists():
            session_id = client.create_session()
            client.call_tool("start_game", {"experiment_name": experiment_name, "game_id": int(game_id)})
        else:
            trace_lines.append("host_game_abort_failed: game was started but no open session marker exists\n")
            return None

        result = client.call_tool("abort_game", {"reason": reason})
        trace_lines.append("host_game_abort_submitted: "
                           f"session_id={client.session_id} done={result.get('done')}\n")
        return result if isinstance(result, dict) else None
    except Exception as error:
        trace_lines.append("host_game_abort_failed: "
                           f"session_id={client.session_id} error={error}\n")
        return None
    finally:
        if client.session_id is not None:
            session_id = client.session_id

            try:
                client.close_session()
                trace_lines.append(f"openenv_session_closed: {session_id}\n")
            except Exception as error:
                trace_lines.append("openenv_session_close_failed: "
                                   f"session_id={session_id} error={error}\n")


# main functions running run_pipeline.py


def load_game_instances(game_name: str, instances_filename: str | None = None,
                        experiment_name: str | None = None) -> GameInstances:
    """Load game instances and optionally restrict them to one experiment.

    Args:
        game_name: Name of the game whose instances should be loaded.
        instances_filename: Optional alternative instances filename.
        experiment_name: Optional experiment name used to filter the instances.

    Returns:
        The loaded and optionally filtered game instances.
    """

    # load the game definition from the clembench registry
    game_registry = GameRegistry.from_directories_and_cwd_files()
    game_spec = game_registry.get_game_spec(game_name)

    # use a custom instances file when one was provided
    if instances_filename:
        game_spec.instances = instances_filename

    # load all instances defined by the game specification
    game_instances = GameInstances.from_game_spec(game_spec)

    # restrict the run to one experiment when requested
    if experiment_name:
        game_instances = game_instances.filter(lambda row: row["experiment"]["name"] == experiment_name)

    return game_instances


def start_server(game_name: str, agent_name: str, agent_player: str, env_agent_models: list[str], gen_args: dict | None,
                 instances_filename: str | None, results_dir: str, completion_path: Path | None = None) -> Process:
    """Start the clembench MCP server and wait until it accepts connections

    Args:
        game_name: Name of the game served by the MCP server
        agent_name: Name of the external agent
        agent_player: Player slot controlled by the external agent
        env_agent_models: Native models assigned to the remaining player slots
        gen_args: Optional generation arguments for native models
        instances_filename: Optional alternative game instances filename
        results_dir: Root directory where clembench writes results
        completion_path: Host-visible file written when an episode reaches done=true

    Returns:
        The running MCP server process
    """

    # ensure that another server is not already using the configured port
    # should not happen anymore after pipeline automatically closes server
    try:
        with socket.create_connection(("127.0.0.1", SERVER_PORT), timeout=1):
            raise RuntimeError(f"Port {SERVER_PORT} is already in use. "
                               "Stop the existing MCP server before running the pipeline.")
    except OSError:
        pass

    # receive concrete startup state or a child traceback during startup
    startup_status_queue = ProcessQueue()
    startup_statuses: list[dict] = []
    server_kwargs = {"game_name": game_name,
                     "agent_name": agent_name,
                     "registry_path": REGISTRY_PATH,
                     "learner_agent": agent_player,
                     "env_agents": _env_agents_from_models(models=env_agent_models, learner_agent=agent_player, game_name=game_name),
                     "gen_args": gen_args,
                     "game_instance_split": None,
                     "instances_filename": instances_filename,
                     "single_pass": False,
                     "results_dir": results_dir,
                     "completion_path": completion_path,
                     "port": SERVER_PORT,
                     "quiet": True}

    # start the mcp server in a separate process
    process = Process(target=_run_mcp_server_process, kwargs={"server_kwargs": server_kwargs,
                                                              "startup_status_queue": startup_status_queue})
    # start the process
    process.start()

    # wait until the server accepts connections
    deadline = time.monotonic() + 30

    while time.monotonic() < deadline:
        _drain_server_startup_status(startup_status_queue, startup_statuses)

        if not process.is_alive():
            process.join(timeout=1)
            time.sleep(0.05)
            _drain_server_startup_status(startup_status_queue, startup_statuses)
            raise RuntimeError("MCP server process exited during startup. "
                               f"Exit code: {process.exitcode}. "
                               f"{_format_server_startup_status(startup_statuses)}")

        try:
            with socket.create_connection(("127.0.0.1", SERVER_PORT), timeout=1):
                return process
        except OSError:
            time.sleep(0.25)

    # stop the process when startup fails
    process.terminate()
    process.join(timeout=5)
    _drain_server_startup_status(startup_status_queue, startup_statuses)

    raise TimeoutError(f"MCP server did not become available on port {SERVER_PORT}. "
                       f"{_format_server_startup_status(startup_statuses)}")


def run_docker_episode(experiment_name: str, game_id: int | str, agent_name: str,
                       model_connection_path: Path | None = None, shared_state_dir: Path | None = None,
                       episode_timeout: float = 600.0,
                       artifact_finalization_timeout: float = ARTIFACT_FINALIZATION_TIMEOUT,
                       artifact_exit_grace: float = ARTIFACT_EXIT_GRACE,
                       registry_path: Path | None = None) -> tuple[str, int, dict]:
    """Run one external-agent episode inside the Docker sandbox.

        Args:
            experiment_name: Name of the experiment containing the game instance.
            game_id: Identifier of the game instance to run.
            agent_name: Name of the external agent from the agent registry.
            model_connection_path: Optional path to the resolved model connection file.
            shared_state_dir: Optional directory shared between the host and container.
            episode_timeout: Maximum wall-clock seconds allowed for the complete episode.
            artifact_finalization_timeout: Maximum seconds allowed for adapter
                finalization after the game result is frozen.
            artifact_exit_grace: Seconds allowed for natural container exit after
                the artifact-ready marker is observed.
            registry_path: Optional isolated run registry passed unchanged to the container.

        Returns:
            The container output, return code, and host episode metadata.
    """

    if episode_timeout <= 0:
        raise ValueError("episode_timeout must be greater than zero seconds")

    if artifact_finalization_timeout <= 0:
        raise ValueError("artifact_finalization_timeout must be greater than zero seconds")

    if artifact_exit_grace < 0:
        raise ValueError("artifact_exit_grace must not be negative")

    # verify that the files required by the container exist
    if not KEYS_PATH.exists():
        raise FileNotFoundError(f"Missing credentials file: {KEYS_PATH}")

    registry_path = registry_path or REGISTRY_PATH
    if not registry_path.exists():
        raise FileNotFoundError(f"Missing agent registry: {registry_path}")

    # load the credentials used by external-agent command line tools
    try:
        keys = json.loads(KEYS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid credentials file: {KEYS_PATH}") from error

    if not isinstance(keys, dict):
        raise ValueError(f"Credentials file must contain a JSON object: {KEYS_PATH}")

    container_environment = {"OPENAI_API_KEY": keys.get("openai", {}).get("api_key"),
                             "ANTHROPIC_API_KEY": keys.get("anthropic", {}).get("api_key")}

    container_environment = {name: value for name, value in container_environment.items() if value}

    # build the base docker command and provide the episode information
    command = ["docker", "run", "--rm", "-i", "-e", f"OPENENV_MCP_URL={OPENENV_MCP_URL}", "-e",
               f"GAME_EXPERIMENT={experiment_name}", "-e", f"GAME_INSTANCE_ID={game_id}", "-e", f"AGENT_NAME={agent_name}",
               "-e", "GAME_OBSERVATION_DIR=/workspace/game_observations", "-v",
               f"{PACKAGE_ROOT}:/opt/clemagents/src/clemagents:ro", "-v", f"{SANDBOX_DIR}:/app:ro", "-v",
               f"{registry_path}:/tmp/agent_registry.json:ro"]

    # docker engine on linux needs an explicit route back to the host
    if sys.platform == "linux":
        command.extend(["--add-host", "host.docker.internal:host-gateway"])

    # forward credentials without writing their values into the docker command
    for name in container_environment:
        command.extend(["-e", name])

    completion_path = None
    artifacts_ready_path = None
    artifacts_started_path = None
    cidfile_path = None
    live_trace_path = None

    # mount the writable directory used to exchange runtime state
    if shared_state_dir is not None:
        completion_path = shared_state_dir / "game_completion.json"
        artifacts_ready_path = shared_state_dir / "artifacts_ready.json"
        artifacts_started_path = shared_state_dir / "artifacts_started.json"
        cidfile_path = shared_state_dir / "container.cid"
        live_trace_path = shared_state_dir / "live_agent_trace.log"
        cidfile_path.unlink(missing_ok=True)
        live_trace_path.unlink(missing_ok=True)
        artifacts_ready_path.unlink(missing_ok=True)
        artifacts_started_path.unlink(missing_ok=True)
        command.extend(["--cidfile", str(cidfile_path)])
        command.extend(["-e", "GAME_SESSION_PATH=/run/game-agent/openenv_session.json", "-e",
                        "GAME_STARTED_PATH=/run/game-agent/game_started.json", "-e",
                        "AGENT_ARTIFACT_ROOT=/run/game-agent/artifacts", "-e",
                        "AGENT_ARTIFACTS_READY_PATH=/run/game-agent/artifacts_ready.json", "-e",
                        "AGENT_ARTIFACTS_STARTED_PATH=/run/game-agent/artifacts_started.json", "-e",
                        "AGENT_LIVE_TRACE_PATH=/run/game-agent/live_agent_trace.log", "-v",
                        f"{shared_state_dir}:/run/game-agent:rw"])

    # tell the container where to find the resolved model connection
    if model_connection_path is not None:
        command.extend(["-e", "AGENT_MODEL_CONNECTION_PATH=/run/game-agent/model_connection.json"])

        #  mount the model connection directly when no shared directory is used
        if shared_state_dir is None:
            command.extend(["-v", f"{model_connection_path}:/run/game-agent/model_connection.json:ro"])

    # run the container-side python entrypoint, i.e. run_agent_container
    command.extend([DOCKER_IMAGE, "python", "/app/run_agent_container.py"])

    # collect the container output for the trace
    trace_lines: list[str] = []
    host_completion: dict | None = None
    process = None
    return_code = 1
    episode_started_at = time.monotonic()
    timed_out = False
    terminal_reason = None
    completion_observed_at = None
    artifacts_ready_observed_at = None
    artifacts_ready_marker = None
    artifact_finalization_timed_out = False

    try:
        process_environment = {**os.environ, **container_environment}
        # start the container and combine stdout and stderr
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   encoding="utf-8", errors="replace", bufsize=1, env=process_environment)

        assert process.stdout is not None

        # read stdout in a background thread so the host can watch completion
        # even while a harness is silent or still generating
        output_queue: queue.Queue[str | None] = queue.Queue()

        def collect_output() -> None:
            try:
                for line in process.stdout:
                    output_queue.put(line)
            except (OSError, UnicodeError) as error:
                output_queue.put("host_output_capture_error: "
                                 f"{type(error).__name__}: {error}\n")
            finally:
                output_queue.put(None)

        output_thread = threading.Thread(target=collect_output, daemon=True)
        output_thread.start()
        output_closed = False

        while process.poll() is None or not output_closed:
            try:
                line = output_queue.get(timeout=0.1)
            except queue.Empty:
                line = None
            else:
                if line is None:
                    output_closed = True
                else:
                    trace_lines.append(line)

            if (host_completion is None and completion_path is not None and completion_path.exists()):
                try:
                    completion = json.loads(completion_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    completion = None

                if isinstance(completion, dict) and completion.get("done") is True:
                    host_completion = completion
                    completion_metadata = completion.get("metadata") or {}
                    terminal_reason = ("game_abort"
                                       if completion_metadata.get("control_failure") is True else "game_done")
                    trace_lines.append("host_game_completion_observed: "
                                       f"source={completion.get('source', 'unknown')}\n")
                    completion_observed_at = time.monotonic()

            if host_completion is not None and artifacts_ready_marker is None:
                ready_marker = _read_json_marker(artifacts_ready_path)

                if isinstance(ready_marker, dict):
                    artifacts_ready_marker = ready_marker
                    artifacts_ready_observed_at = time.monotonic()
                    trace_lines.append("host_agent_artifacts_ready_observed: "
                                       f"status={ready_marker.get('status', 'unknown')}\n")

            now = time.monotonic()

            if (host_completion is not None and artifacts_ready_marker is None and completion_observed_at is not None
                    and now - completion_observed_at >= artifact_finalization_timeout):
                artifact_finalization_timed_out = True
                trace_lines.append("host_artifact_finalization_timeout: "
                                   f"limit_seconds={artifact_finalization_timeout:g}\n")
                _stop_docker_container(process, cidfile_path)

            if (artifacts_ready_marker is not None and artifacts_ready_observed_at is not None
                    and process.poll() is None and now - artifacts_ready_observed_at >= artifact_exit_grace):
                trace_lines.append("host_container_exit_grace_expired_after_artifacts_ready: "
                                   f"limit_seconds={artifact_exit_grace:g}\n")
                _stop_docker_container(process, cidfile_path)

            elapsed = now - episode_started_at

            if (host_completion is None and not timed_out and elapsed >= episode_timeout):
                timed_out = True
                terminal_reason = "episode_timeout"
                trace_lines.append("host_episode_timeout: "
                                   f"limit_seconds={episode_timeout:g} elapsed_seconds={elapsed:.3f}\n")

                # stop inference first; preserving partial artifacts does not extend play
                _stop_docker_container(process, cidfile_path)
                host_completion = _abort_unfinished_episode(shared_state_dir=shared_state_dir,
                                                            experiment_name=experiment_name, game_id=game_id,
                                                            trace_lines=trace_lines, reason=EPISODE_TIMEOUT_REASON)

        return_code = process.wait()
        output_thread.join(timeout=1)

        if process.stdout is not None:
            process.stdout.close()

        if artifacts_ready_marker is None:
            artifacts_ready_marker = _read_json_marker(artifacts_ready_path)

            if artifacts_ready_marker is not None:
                artifacts_ready_observed_at = time.monotonic()
        if ((timed_out or host_completion is None) and live_trace_path is not None and live_trace_path.exists()):
            partial_trace = live_trace_path.read_text(encoding="utf-8", errors="replace")
            captured_trace = "".join(trace_lines)

            if partial_trace and partial_trace not in captured_trace:
                trace_lines.extend(["partial_agent_trace_snapshot_start\n", partial_trace,
                                    "\npartial_agent_trace_snapshot_end\n"])

        # do not abort the benchmark because one harness exits nonzero
        # cleanup finalizes that episode before the caller moves to the next one
        if return_code != 0 and host_completion is None and not timed_out:
            trace_lines.append(f"agent_container_exit: return_code={return_code}\n")

    finally:
        # recover an unfinished openenv session
        if shared_state_dir is not None:
            completion_path = shared_state_dir / "game_completion.json"

            if host_completion is None and completion_path.exists():
                try:
                    completion = json.loads(completion_path.read_text(encoding="utf-8"))
                    host_completion = completion if isinstance(completion, dict) else None
                except (OSError, json.JSONDecodeError):
                    host_completion = None

            completed = bool(host_completion and host_completion.get("done") is True)

            if not completed:
                terminal_reason = terminal_reason or "agent_exit_before_game_done"
                host_completion = _abort_unfinished_episode(shared_state_dir=shared_state_dir,
                                                            experiment_name=experiment_name, game_id=game_id,
                                                            trace_lines=trace_lines, reason=AGENT_TERMINATION_REASON)

    container_elapsed_seconds = time.monotonic() - episode_started_at
    episode_elapsed_seconds = (completion_observed_at -
                               episode_started_at if completion_observed_at is not None else container_elapsed_seconds)
    artifact_finalization_seconds = (artifacts_ready_observed_at - completion_observed_at if
                                     (artifacts_ready_observed_at is not None
                                      and completion_observed_at is not None) else None)
    artifact_directory = _shared_artifact_directory(shared_state_dir, artifacts_ready_marker)
    finalized = bool(artifact_directory is not None and artifacts_ready_marker.get("ready") is True
                     and artifacts_ready_marker.get("status") == "complete")
    if artifact_directory is None:
        artifact_directory = _shared_artifact_directory(shared_state_dir, _read_json_marker(artifacts_started_path))
    artifact_capture_status = "complete" if finalized else "partial" if artifact_directory is not None else "unavailable"
    episode_metadata = {"episode_timeout_seconds":
                        episode_timeout,
                        "episode_timed_out":
                        timed_out,
                        "episode_elapsed_seconds":
                        episode_elapsed_seconds,
                        "container_elapsed_seconds":
                        container_elapsed_seconds,
                        "episode_terminal_reason":
                        terminal_reason or "container_exit",
                        "host_completion":
                        host_completion,
                        "artifact_finalization_timeout_seconds":
                        artifact_finalization_timeout,
                        "artifact_finalization_timed_out":
                        artifact_finalization_timed_out,
                        "artifact_finalization_seconds":
                        artifact_finalization_seconds,
                        "artifacts_ready":
                        bool(artifacts_ready_marker and artifacts_ready_marker.get("ready") is True),
                        "artifacts_ready_status":
                        (artifacts_ready_marker.get("status") if artifacts_ready_marker is not None else "unavailable"),
                        "artifact_capture_status": artifact_capture_status,
                        "agent_artifact_relative_path": (str(artifact_directory.relative_to(shared_state_dir.resolve()))
                                                         if artifact_directory is not None and shared_state_dir is not None else None)}
    return "".join(trace_lines), return_code, episode_metadata


def write_agent_artifacts(results_dir: str, run_dir: str, game_name: str, experiment_name: str, game_id: int | str,
                          before_instance_dirs: set[Path], trace_text: str, metadata: dict,
                          source_artifact_dir: str | Path | None = None,
                          registry_path: str | Path = REGISTRY_PATH) -> Path:
    """Write agent artifacts into the corresponding clembench instance directory.

    Args:
        results_dir: Root directory containing the clembench results.
        run_dir: Exact dialogue-pair directory used by the MCP server.
        game_name: Name of the game that was run.
        experiment_name: Name of the experiment containing the episode.
        game_id: Identifier of the game instance that was run.
        before_instance_dirs: Instance directories that existed before the run.
        trace_text: Raw output captured from the agent container
        metadata: Additional information to store alongside the trace.
        source_artifact_dir: Optional finalized or interrupted adapter-artifact directory.
        registry_path: Agent registry used to resolve the harness parser

    Returns:
        Path to the written agent trace file.
    """

    # find all instance directories that exist after the run
    results_path = Path(results_dir)
    after_instance_dirs = {path for path in results_path.glob(f"{run_dir}/{game_name}/{experiment_name}/instance_*") if path.is_dir()}

    try:
        run_started_timestamp = datetime.fromisoformat(str(metadata["started_at"])).timestamp()
    except (KeyError, TypeError, ValueError):
        # if the caller cannot establish the run boundary, prefer the failure
        # tree over risking an overwrite of an unrelated historical episode
        run_started_timestamp = datetime.now(timezone.utc).timestamp()

    # record the modification time of every instance directory
    instance_timestamps = {}

    for instance_dir in after_instance_dirs:
        timestamp_candidates = [instance_dir / "interactions.json", instance_dir / "instance.json", instance_dir]

        instance_timestamps[instance_dir] = max(path.stat().st_mtime for path in timestamp_candidates if path.exists())

    # first try to identify an instance directory created by this run
    new_instance_dirs = after_instance_dirs - before_instance_dirs

    if new_instance_dirs:
        output_dir = max(new_instance_dirs, key=instance_timestamps.get)

    else:
        # only reuse an existing instance directory if this run actually
        # modified it. a failed agent may create no instance at all
        matching_instance_dirs = []

        for instance_dir in after_instance_dirs:
            instance_path = instance_dir / "instance.json"

            if not instance_path.exists():
                continue

            try:
                instance = json.loads(instance_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue

            if (int(instance.get("game_id")) == int(game_id)
                    and instance_timestamps[instance_dir] >= run_started_timestamp):
                matching_instance_dirs.append(instance_dir)

        if matching_instance_dirs:
            output_dir = max(matching_instance_dirs, key=instance_timestamps.get)

        else:
            # preserve the trace separately when clembench created no episode
            output_dir = (results_path / "_agent_failures" / run_dir / game_name / experiment_name /
                          f"game_id_{game_id}" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
            output_dir.mkdir(parents=True, exist_ok=True)

    # preserve each native artifact set under its own run directory
    # keeping it separate prevents either a harness artifact or a rerun from
    # overwriting clembench's game records or a previous native trace
    copied_artifact_dir = None

    if source_artifact_dir is not None:
        source_path = Path(source_artifact_dir)

        if source_path.is_dir():
            artifact_root = output_dir / "agent_artifacts"
            artifact_root.mkdir(parents=True, exist_ok=True)
            copied_artifact_dir = artifact_root / source_path.name
            suffix = 1

            while copied_artifact_dir.exists():
                copied_artifact_dir = artifact_root / f"{source_path.name}_{suffix}"
                suffix += 1

            # native artifacts can contain links into the container's installed
            # runtime. preserve those links verbatim: following them from the
            # host can fail after the container has exited, and would copy data
            # that was never part of the finalized artifact set
            shutil.copytree(source_path, copied_artifact_dir, symlinks=True)
            metadata["native_artifact_directory"] = str(copied_artifact_dir.relative_to(output_dir))
            # interrupted adapters may not have assembled their console log yet
            artifact_trace = copied_artifact_dir / "agent_trace.log"
            if not artifact_trace.exists() and not artifact_trace.is_symlink():
                artifact_trace.write_text(trace_text, encoding="utf-8")

    # write the raw host trace and its metadata into the selected directory
    trace_path = output_dir / "agent_trace.log"
    trace_path.write_text(trace_text, encoding="utf-8")

    metadata_path = output_dir / "agent_trace_meta.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    agent_name = metadata.get("agent")

    if isinstance(agent_name, str) and agent_name:
        try:
            harness_class = harness_class_for_agent(agent_name, registry_path)
            harness_class.serialize_standardized_agent_trace(episode_dir=output_dir, artifact_dir=copied_artifact_dir,
                                                             metadata=metadata)
        except Exception as error:
            agent_loop_path = output_dir / "agent_loop.json"
            failure = missing_agent_trace(f"Trace parsing failed for {agent_name}: {type(error).__name__}: {error}")
            failure = normalize_agent_trace({**failure, "metadata": metadata})
            agent_loop_path.write_text(json.dumps(failure, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return trace_path


def append_agent_episode_outcome(results_dir: str | Path, metadata: dict) -> Path:
    """Append one host-observed episode outcome for aggregate diagnostics."""
    output_path = Path(results_dir) / "agent_episode_outcomes.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    record = {"agent": metadata.get("agent"),
              "game": metadata.get("game"),
              "experiment_name": metadata.get("experiment_name"),
              "game_id": metadata.get("game_id"),
              "started_at": metadata.get("started_at"),
              "finished_at": metadata.get("finished_at"),
              "episode_elapsed_seconds": metadata.get("episode_elapsed_seconds"),
              "episode_timeout_seconds": metadata.get("episode_timeout_seconds"),
              "episode_timed_out": metadata.get("episode_timed_out"),
              "episode_terminal_reason": metadata.get("episode_terminal_reason"),
              "artifact_finalization_seconds": metadata.get("artifact_finalization_seconds"),
              "artifact_finalization_timed_out": metadata.get("artifact_finalization_timed_out"),
              "artifacts_ready": metadata.get("artifacts_ready"),
              "artifact_capture_status": metadata.get("artifact_capture_status"),
              "return_code": metadata.get("return_code")}

    with output_path.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    return output_path


# helper functions


def _player_ids_from_clemgame(game_name: str) -> list[str] | None:
    """Return player ids declared by the game's clemgame.json, if available."""
    metadata_path = Path.cwd() / game_name / "clemgame.json"
    if not metadata_path.exists():
        return None

    data = json.loads(metadata_path.read_text(encoding="utf-8"))

    specs: list[dict] = []
    if isinstance(data, list):
        specs = [item for item in data if isinstance(item, dict)]
    elif isinstance(data, dict):
        if data.get("name") == game_name or data.get("game_name") == game_name:
            specs.append(data)
        for key in ("games", "game_specs", "benchmarks"):
            value = data.get(key)
            if isinstance(value, list):
                specs.extend(item for item in value if isinstance(item, dict))

    for spec in specs:
        if spec.get("name") != game_name and spec.get("game_name") != game_name:
            continue

        player_count = (spec.get("players") or spec.get("num_players") or spec.get("n_players")
                        or spec.get("number_of_players"))

        if isinstance(player_count, int):
            return [f"player_{index}" for index in range(player_count)]

        roles = spec.get("roles")
        if isinstance(roles, list):
            return [f"player_{index}" for index in range(len(roles))]

    return None


def _env_agents_from_models(models: list[str], learner_agent: str, game_name: str | None = None) -> dict[str, str]:
    """Assign native models to all players except the external agent.

    Args:
        models: Native clembench models supplied for non-agent players.
        learner_agent: Player slot controlled by the external agent.
        game_name: Optional game name used to determine the declared player count.

    Returns:
        A mapping from non-agent player IDs to native model names.

    Raises:
        ValueError: If the external-agent player slot is invalid or too few
            native models were supplied.
    """

    # read player ids from the game metadata when available
    player_ids = _player_ids_from_clemgame(game_name) if game_name else None

    # fall back to inferring one non-agent player per supplied model
    if player_ids is None:
        player_ids = [f"player_{index}" for index in range(len(models) + 1)]

    if learner_agent not in player_ids:
        raise ValueError(f"Cannot use learner_agent={learner_agent!r}. "
                         f"Valid choices are: {player_ids}")

    # remove the player controlled by the external agent
    env_player_ids = [player_id for player_id in player_ids if player_id != learner_agent]

    if len(models) < len(env_player_ids):
        raise ValueError(f"Need {len(env_player_ids)} native model(s) for "
                         f"{env_player_ids}, but got {len(models)}: {models}")

    return dict(zip(env_player_ids, models))


def _write_run_registry(agent_name: str, output_dir: Path, agent_settings: dict) -> Path:
    """Copy registry settings for a run without editing the user's registry.

    Args:
        agent_name: selected registry agent
        output_dir: temporary run directory
        agent_settings: explicit CLI settings passed unchanged to the adapter

    Returns:
        path to the isolated run registry
    """
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    entry = next(entry for entry in registry if entry.get("agent_name") == agent_name)
    config = {**entry.get("agent_config", {}), **agent_settings}
    adapter = harness_class_for_agent(agent_name, REGISTRY_PATH)
    try:
        inspect.signature(adapter).bind(**config)
    except TypeError as error:
        raise ValueError(f"{adapter.__name__} does not accept these agent settings: {error}. "
                         "Only adapter-supported native controls can be used; no API fields are injected") from error
    entry["agent_config"] = config
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "agent_registry.json"
    path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    path.chmod(0o600)
    return path


def _write_agent_model_connection(agent_name: str, output_dir: Path,
                                  registry_path: Path | None = None) -> Path | None:
    """Resolve and write the model connection required by an external agent.

    Args:
        agent_name: name of the external agent from the agent registry
        output_dir: directory where the connection file should be written
        registry_path: optional isolated run registry

    Returns:
        path to the written model connection file, or None when the agent
        does not use a clemcore model connection
    """

    # resolve the clemcore model used by the external agent
    connection = resolve_agent_model_connection(agent_name=agent_name, registry_path=registry_path or REGISTRY_PATH)

    # agents without an associated clemcore model need no connection file
    if connection is None:
        return None

    # make a model server on the host reachable from docker
    # this is relevant when endpoint is on local machine, e.g. through ollama)
    base_url = connection.get("base_url")

    if isinstance(base_url, str):
        parsed_url = urlsplit(base_url)

        if parsed_url.hostname in {"127.0.0.1", "localhost", "::1"}:
            port_suffix = f":{parsed_url.port}" if parsed_url.port else ""
            container_base_url = urlunsplit((parsed_url.scheme, f"host.docker.internal{port_suffix}", parsed_url.path,
                                             parsed_url.query, parsed_url.fragment,
                                             ))
            connection["base_url"] = container_base_url

    # write the connection into the temporary directory shared with the container
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model_connection.json"
    output_path.write_text(json.dumps(connection, indent=2), encoding="utf-8")

    # restrict access because the connection may contain credentials
    output_path.chmod(0o600)

    # print the resolved connection without exposing sensitive values
    print("agent_model_connection: "
          f"clem_model={connection.get('clem_model')} "
          f"backend={connection.get('backend')} "
          f"runtime_model={connection.get('model')}")

    return output_path
