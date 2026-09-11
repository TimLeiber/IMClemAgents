import argparse
import tempfile
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path

from tqdm import tqdm

from clemagents.mcp.server import result_run_dir_name

# import all functions from utils needed to run main
from .utils import (
    DOCKER_IMAGE, # Docker image used for agent episodes
    REGISTRY_PATH, # external-agent registry used by server and container
    _env_agents_from_models, # map native models to player slots
    _write_agent_model_connection, # write the agent model configuration

    append_agent_episode_outcome, # append aggregate episode lifecycle metadata
    load_game_instances, # load and filter game instances
    run_docker_episode, # run one agent episode in Docker
    start_server, # start the MCP server
    write_agent_artifacts # store the captured agent artifacts
)


def main() -> None:
    # define the command line interface
    parser = argparse.ArgumentParser(
        description="Run external agent harnesses through clembench/OpenEnv/MCP.",
    )

    # define the pipeline arguments
    parser.add_argument("-g",
                        "--game",
                        required=True,
                        help="Game name, matching clem run -g.")
    parser.add_argument("-a",
                        "--agent",
                        required=True,
                        help="External agent name from the agent registry.")
    parser.add_argument("-m",
                        "--models",
                        nargs="+",
                        default=[],
                        help="Native clem model(s) for non-external players, matching clem run -m style.")
    parser.add_argument("--agent-player",
                        default="player_0",
                        help="Player slot controlled by the external agent, e.g. player_0 or player_1.")
    parser.add_argument("-e",
                        "--experiment_name",
                        default=None,
                        help="Optional experiment filter, matching clem run -e.")
    parser.add_argument("-i",
                        "--instances_filename",
                        default=None,
                        help="Instances file name without .json, matching clem run -i.")
    parser.add_argument("-r",
                        "--results_dir",
                        default="results/external-agents",
                        help="Results root directory, matching clem run -r style.")
    parser.add_argument("-t",
                        "--temperature",
                        type=float,
                        default=None,
                        help=("Optional sampling temperature for the external model and "
                              "native environment models. Omitted preserves the external "
                              "provider default and uses 0.0 for native models."))
    parser.add_argument("-l",
                        "--max_tokens",
                        type=int,
                        default=300,
                        help="Maximum output tokens for native clem models, matching clem run -l.")
    parser.add_argument("--max-instances",
                        type=int,
                        default=None,
                        help="Optional debugging limit for number of selected instances to run.")
    parser.add_argument("--episode-timeout",
                        type=float,
                        default=600.0,
                        help="Host-enforced wall-clock limit in seconds for every agent episode.")

    # parse the command line arguments
    args = parser.parse_args()

    if args.episode_timeout <= 0:
        parser.error("--episode-timeout must be greater than zero seconds")

    native_temperature = args.temperature if args.temperature is not None else 0.0
    # map native models to non-agent player slots
    env_agents = _env_agents_from_models(
        models=args.models,
        learner_agent=args.agent_player,
        game_name=args.game,
    )

    # ----- main step 1 -----
    # load and filter the selected game instances
    game_instances = load_game_instances(
        game_name=args.game,
        instances_filename=args.instances_filename,
        experiment_name=args.experiment_name,
    )

    # ensure that at least one instance was selected
    total = len(game_instances)

    if total == 0:
        raise RuntimeError("No game instances selected.")

    # print the resolved pipeline configuration
    print(f"game: {args.game}")
    print(f"agent: {args.agent}")
    print(f"agent_player: {args.agent_player}")
    print(f"models: {args.models}")
    print(f"env_agents: {env_agents}")
    print(f"instances_filename: {args.instances_filename or 'instances'}")
    print(f"experiment_name: {args.experiment_name or '<all>'}")
    print(f"results_dir: {args.results_dir}")
    print(f"temperature: {args.temperature if args.temperature is not None else '<provider default>'}")
    print(f"episode_timeout: {args.episode_timeout:g} seconds")
    print(game_instances.describe())

    result_run_dir = result_run_dir_name(
        agent_name=args.agent,
        registry_path=REGISTRY_PATH,
        learner_agent=args.agent_player,
        env_agents=env_agents,
    )

    # create temporary shared storage for the agent runtime
    temp_dir = tempfile.TemporaryDirectory(prefix="clem-agent-model-")
    # resolve and write the external agent model connection
    model_connection_path = _write_agent_model_connection(
        agent_name=args.agent,
        output_dir=Path(temp_dir.name),
        temperature=args.temperature,
    )
    game_completion_path = Path(temp_dir.name) / "game_completion.json"

    # ---- main step 2 -----
    # start the MCP game server, exposing game as tools for the agent
    server_process = start_server(
        game_name=args.game,
        agent_name=args.agent,
        agent_player=args.agent_player,
        env_agent_models=args.models,
        gen_args={
            "temperature": native_temperature,
            "max_tokens": args.max_tokens,
        },
        instances_filename=args.instances_filename,
        results_dir=args.results_dir,
        completion_path=game_completion_path,
    )

    # try block attempts to run external agent in docker container on each episode
    try:
        # apply the optional debugging instance limit
        selected_instances = game_instances

        if args.max_instances is not None:
            selected_instances = list(islice(game_instances, args.max_instances))

        # run each selected game instance
        for row in tqdm(selected_instances, desc="Playing game instances"):
            experiment = row["experiment"]
            game_instance = row["game_instance"]

            experiment_name = experiment["name"]
            game_id = game_instance["game_id"]

            # record the episode directories that exist before this run
            before_instance_dirs = {path
                                    for path in Path(args.results_dir).glob(
                                        f"{result_run_dir}/{args.game}/{experiment_name}/instance_*"
                                    )
                                    if path.is_dir()}

            # record start time and remove session file from previous episode
            started_at = datetime.now(timezone.utc)
            openenv_session_path = Path(temp_dir.name) / "openenv_session.json"
            game_started_path = Path(temp_dir.name) / "game_started.json"

            openenv_session_path.unlink(missing_ok=True)
            game_started_path.unlink(missing_ok=True)
            game_completion_path.unlink(missing_ok=True)

            # ----- main step 3 -----
            # run the external agent inside Docker
            trace_text, return_code, runtime_metadata = run_docker_episode(
                experiment_name=experiment_name,
                game_id=game_id,
                agent_name=args.agent,
                model_connection_path=model_connection_path,
                shared_state_dir=Path(temp_dir.name),
                episode_timeout=args.episode_timeout,
            )
            artifact_relative_path = runtime_metadata.get(
                "agent_artifact_relative_path"
            )
            source_artifact_dir = (
                Path(temp_dir.name) / artifact_relative_path
                if isinstance(artifact_relative_path, str)
                else None
            )
            finished_at = datetime.now(timezone.utc)
            metadata = {
                "agent": args.agent,
                "agent_player": args.agent_player,
                "models": args.models,
                "env_agents": env_agents,
                "game": args.game,
                "experiment_name": experiment_name,
                "game_id": game_id,
                "docker_image": DOCKER_IMAGE,
                "return_code": return_code,
                "started_at": started_at.isoformat(),
                "finished_at": finished_at.isoformat(),
                **runtime_metadata,
            }
            # ----- main step 4 -----
            # write the captured trace, metadata, and standardized agent loop
            write_agent_artifacts(
                results_dir=args.results_dir,
                run_dir=result_run_dir,
                game_name=args.game,
                experiment_name=experiment_name,
                game_id=game_id,
                before_instance_dirs=before_instance_dirs,
                trace_text=trace_text,
                metadata=metadata,
                source_artifact_dir=source_artifact_dir,
            )
            append_agent_episode_outcome(args.results_dir, metadata)

    # always terminate/kill the running server
    # (if this fails the server is blocked and will not allow to be called for another game to be played)
    finally:

        # stop the MCP server
        if server_process.is_alive():
            server_process.terminate()
            server_process.join(timeout=5)

        # fallback to kill server process if process fails to exit initially
        if server_process.is_alive():
            server_process.kill()
            server_process.join(timeout=5)

        temp_dir.cleanup()


if __name__ == "__main__":
    main()
