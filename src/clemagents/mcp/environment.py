import base64
import json
import logging
import mimetypes
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict
from urllib.parse import unquote_to_bytes, urlparse

import requests
from datasets import load_dataset
from fastmcp import FastMCP
from pettingzoo import AECEnv
from pettingzoo.utils.wrappers import OrderEnforcingWrapper

from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.mcp_types import Action, Observation

from clemcore.backends import load_models
from clemcore.clemgame.callbacks.base import GameBenchmarkCallbackList
from clemcore.clemgame.envs.openenv.models import (ClemGameAction, ClemGameObservation, ClemGameState)
from clemcore.clemgame.envs.pettingzoo import check_agent_mapping_for_training
from clemcore.clemgame.envs.pettingzoo.master import GameMasterEnv
from clemcore.clemgame.envs.pettingzoo.wrappers import (GameBenchmarkWrapper, GameInstanceIteratorWrapper,
                                                        SinglePlayerWrapper)
from clemcore.clemgame.instances import GameInstances, to_instance_filter
from clemcore.clemgame.master import GameState
from clemcore.clemgame.registry import GameRegistry

module_logger = logging.getLogger(__name__)


def _image_mime_type(reference: str, content_type: str | None = None) -> str:
    """Resolve an image MIME type from an HTTP header or source name."""
    if content_type:
        mime_type = content_type.split(";", 1)[0].strip().lower()

        if mime_type.startswith("image/"):
            return mime_type

    path = urlparse(reference).path if reference.startswith(("http://", "https://")) else reference
    mime_type, _encoding = mimetypes.guess_type(path)

    if mime_type and mime_type.startswith("image/"):
        return mime_type

    raise ValueError(f"Cannot determine image MIME type for {reference!r}")


def _portable_image(image: Any) -> dict[str, str]:
    """Encode one game image for transport across the host/container boundary."""
    if isinstance(image, dict):
        data = image.get("data")
        mime_type = image.get("mimeType")

        if isinstance(data, str) and isinstance(mime_type, str):
            return {"data": data, "mimeType": mime_type}

        raise ValueError("Encoded images require string data and mimeType fields")

    reference = os.fspath(image)

    if reference.startswith("data:"):
        header, separator, payload = reference.partition(",")

        if not separator:
            raise ValueError("Malformed image data URL")

        media_type = header[5:].split(";", 1)[0].strip().lower()

        if not media_type.startswith("image/"):
            raise ValueError(f"Unsupported data URL media type: {media_type!r}")

        if ";base64" in header.lower():
            encoded = payload
        else:
            encoded = base64.b64encode(unquote_to_bytes(payload)).decode("ascii")

        return {"data": encoded, "mimeType": media_type}

    if reference.startswith(("http://", "https://")):
        response = requests.get(reference, timeout=(30, 120))
        response.raise_for_status()
        mime_type = _image_mime_type(reference, response.headers.get("Content-Type"))
        image_bytes = response.content
    else:
        path = Path(reference)
        mime_type = _image_mime_type(reference)
        image_bytes = path.read_bytes()

    return {"data": base64.b64encode(image_bytes).decode("ascii"), "mimeType": mime_type}


def _portable_context(context: dict) -> dict:
    """Copy a game context and make any image references JSON-portable."""
    portable = deepcopy(context)
    images = portable.get("image")

    if images is None:
        return portable

    if not isinstance(images, (list, tuple)):
        images = [images]

    portable["image"] = [_portable_image(image) for image in images]
    return portable


def _observation_to_dict(observation: ClemGameObservation) -> dict:
    """Convert an environment observation into the payload returned by MCP tools

    Args:
        observation: observation returned by the wrapped clembench environment

    Returns:
        The observation fields as a plain dictionary
    """
    return {"context": _portable_context(observation.context),
            "reward": observation.reward,
            "done": observation.done,
            "metadata": observation.metadata}


def _terminal_details(state: GameState, truncated: bool) -> tuple[str | None, str | None]:
    """Return standardized outcome and reason values for a terminal game state."""
    outcome = getattr(state, "outcome", None)
    outcome = getattr(outcome, "value", outcome)

    if truncated and outcome in {None, "running"}:
        outcome = "truncated"

    outcome = str(outcome) if outcome is not None else None
    error = getattr(state, "error", None)
    reason = getattr(error, "reason", None)

    if reason is None and error is not None:
        reason = str(error)

    reason = str(reason) if reason else None
    return outcome, reason


def _terminal_context(outcome: str | None, reason: str | None) -> dict[str, str]:
    """Create the model-facing message for a terminal step without new context."""
    messages = {"success": "The game is complete. The objective was achieved.",
                "failure": "The game is complete. The objective was not achieved.",
                "aborted": "The game ended before normal completion.",
                "truncated": "The game ended because its execution limit was reached."}
    content = messages.get(outcome, "The game has ended.")

    if reason:
        content += f"\nReason: {reason}"

    return {"role": "user", "content": content}


class SelectableClemGameEnvironment:
    """OpenEnv environment that serves one exactly selected clembench instance.

    This mirrors the outer loop of a regular clembench run without changing
    clemcore.clemgame internals. A caller selects a single instance by passing
    experiment_name and game_id into reset, which rebuilds the underlying
    pettingzoo wrapper stack with a filter matching only that row.
    """

    def __init__(self, game_name: str, *, instances_filename: str | None = None, game_instance_split: str | None = None,
                 single_pass: bool = False, learner_agent: str = "player_0", env_agents: Dict[str, str] | None = None,
                 gen_args: Dict[str, Any] | None = None, callbacks: GameBenchmarkCallbackList | None = None,
                 reward_func: Callable[[dict, str, GameState, dict], float] | None = None,
                 feedback_func: Callable[[dict, str, GameState, dict], str | None] | None = None):
        module_logger.info("Initialize SelectableClemGameEnvironment: "
                           "game_name=%s, instances_filename=%s, game_instance_split=%s, "
                           "learner_agent=%s, env_agents=%s", game_name, instances_filename, game_instance_split, learner_agent,
                           env_agents)

        self._game_name = game_name
        self._instances_filename = instances_filename
        self._game_instance_split = game_instance_split
        self._single_pass = single_pass
        self._learner_agent = learner_agent
        self._callbacks = callbacks
        self._reward_func = reward_func
        self._feedback_func = feedback_func
        self._game_env: AECEnv | None = None
        self._state = ClemGameState(game_name=game_name, episode_id="episode_0", step_count=0, episode_count=0)

        env_agents = env_agents or {}

        # look up the game definition and optionally override its instances file
        game_registry = GameRegistry.from_directories_and_cwd_files()
        self._game_spec = game_registry.get_game_spec(game_name)

        if instances_filename:
            self._game_spec.instances = instances_filename

        # reject player mappings the game does not support
        check_agent_mapping_for_training(self._game_spec, {learner_agent: "learner", **env_agents})

        # single-player games keep the raw names, multi-player games need loaded models
        if self._game_spec.is_multi_player():
            agent_models = load_models(list(env_agents.values()), gen_args)
            self._env_agents = dict(zip(env_agents.keys(), agent_models))
        else:
            self._env_agents = env_agents

    def reset(self, seed=None, episode_id=None, experiment_name: str | None = None, game_id: int | str | None = None,
              **kwargs) -> ClemGameObservation:
        """Build the game environment for the selected instance and start an episode

        Args:
            seed: optional seed forwarded to the wrapped environment
            episode_id: optional explicit episode identifier
            experiment_name: name of the experiment holding the requested instance
            game_id: identifier of the requested instance inside the experiment

        Returns:
            The initial observation of the started episode
        """
        if episode_id is not None:
            kwargs["episode_id"] = episode_id

        # discard the environment of the previous episode before building a new one
        if self._game_env is not None:
            self._game_env.close()

        # reset step 1
        # restrict the instances to the configured split, when a split was configured
        split_filter = None

        if self._game_instance_split:
            dataset = load_dataset("colab-potsdam/playpen-data", "instances", split=self._game_instance_split)
            split_filter = to_instance_filter(dataset)

        # reset step 2
        # narrow the remaining instances down to the single requested one
        if experiment_name is None and game_id is None:
            # without a selection the split filter alone decides which rows are served
            instances_filter = split_filter
        elif experiment_name is None or game_id is None:
            raise ValueError("Both experiment_name and game_id must be provided for exact instance selection.")
        else:
            selected_game_id = int(game_id)

            def instances_filter(row: dict) -> bool:
                if split_filter is not None and not split_filter(row):
                    return False

                return (row["experiment"]["name"] == experiment_name
                        and int(row["game_instance"]["game_id"]) == selected_game_id)

        # reset step 3
        # build the wrapper stack from the inside out, each layer adding one concern

        # the game master runs the game, the benchmark wrapper adds callbacks and scoring
        game_env = GameBenchmarkWrapper(GameMasterEnv, game_spec=self._game_spec, callbacks=self._callbacks,
                                        reward_func=self._reward_func, feedback_func=self._feedback_func)

        # reject calls made in an invalid order, such as stepping before resetting
        game_env = OrderEnforcingWrapper(game_env)

        # collect the instances this episode is allowed to draw from
        game_instances = GameInstances.from_game_spec(self._game_spec)
        game_instances = game_instances.filter(instances_filter)

        if len(game_instances) == 0:
            raise ValueError("No game instances matched the requested selection.")

        # supply those instances to the game, one per episode
        game_env = GameInstanceIteratorWrapper(game_env, game_instances, single_pass=self._single_pass)

        # answer for every native player slot so only the learner slot stays exposed
        game_env = SinglePlayerWrapper(game_env, self._learner_agent, env_agents=self._env_agents)

        self._game_env = game_env

        # reset step 4
        # start the episode and record it in the environment state
        module_logger.info("Reset SelectableClemGameEnvironment '%s' for experiment=%s, game_id=%s",
                           self._state.game_name, experiment_name, game_id)

        options = kwargs if kwargs else None
        self._game_env.reset(seed=seed, options=options)
        observation, _, done, truncated, info = self._read_observation()

        self._state.step_count = 0
        self._state.episode_count += 1
        self._state.episode_id = f"episode_{self._state.episode_count}"

        return ClemGameObservation(context=observation, done=done, metadata={**info, "truncated": truncated})

    def _read_observation(self) -> tuple:
        """Read the learner's turn or terminal state, then finish dead-agent cleanup."""
        env = self._game_env
        if env is None:
            raise RuntimeError("Environment has not been reset. Call start_game first.")

        if env.agent_selection is None:
            # autoplay may already have removed every agent. last() requires a
            # selected agent, but the game can still supply its final context
            agent = self._learner_agent
            return (env.observe(agent), env.rewards.get(agent, 0.0), env.terminations.get(agent, True),
                    env.truncations.get(agent, False), env.infos.get(agent, {}))

        observation, reward, done, truncated, info = env.last()
        if done or truncated:
            # capture the result before pettingzoo removes the learner's state
            # singleplayerwrapper also completes the other players' cleanup
            observation, info = deepcopy(observation), deepcopy(info)
            env.step(None)
        return observation, reward, done, truncated, info

    def step(self, action: ClemGameAction, timeout_s=None, **kwargs) -> ClemGameObservation:
        """Apply one agent response to the running game

        Args:
            action: response submitted by the external agent
            timeout_s: optional timeout accepted for interface compatibility

        Returns:
            The observation following the applied response
        """
        if self._game_env is None:
            raise RuntimeError("Environment has not been reset. Call start_game first.")

        game_master = self._game_env.unwrapped.game_master
        current_player = game_master.current_player
        previous_context = deepcopy(game_master.get_context_for(current_player))
        stored_context = None
        contexts = getattr(game_master, "context_for_player", None)

        if isinstance(contexts, dict) and current_player is not None:
            stored_context = deepcopy(contexts.get(current_player.name))

        self._game_env.step(action.response)
        observation, reward, done, truncated, info = self._read_observation()
        self._state.step_count += 1

        metadata = {**info, **dict(truncated=truncated)}

        if done or truncated:
            outcome, reason = _terminal_details(game_master.state, truncated)
            metadata["outcome"] = outcome

            if reason:
                metadata["terminal_reason"] = reason

            if observation is None or observation in (previous_context, stored_context):
                observation = _terminal_context(outcome, reason)
                metadata["terminal_context_normalized"] = True

        return ClemGameObservation(context=observation, reward=float(reward), done=done, metadata=metadata)

    def abort(self, reason: str) -> ClemGameObservation:
        """Abort the active episode through the normal game finalization path.

        Args:
            reason: Synthetic response recorded as the cause of the abort

        Returns:
            The terminal observation produced by the wrapped game environment
        """
        if self._game_env is None:
            raise RuntimeError("Environment has not been reset. Call start_game first.")

        game_master = self._game_env.unwrapped.game_master
        game_master.state.abort()
        self._game_env.step(reason)
        observation, reward, done, truncated, info = self._read_observation()

        if not done:
            raise RuntimeError("Control abort did not terminate the game episode.")

        return ClemGameObservation(context=observation, reward=float(reward), done=done, metadata={**info,
                                                                                                   **dict(truncated=truncated, control_failure=True)})

    @property
    def state(self) -> ClemGameState:
        return self._state

    def close(self) -> None:
        """Close the wrapped game environment"""
        module_logger.info("Close SelectableClemGameEnvironment %s", self._state.game_name)

        if self._game_env is not None:
            self._game_env.close()
            self._game_env = None


class ClemGameMCPEnvironment(MCPEnvironment):
    """MCP-facing wrapper exposing a clembench game as MCP tools.

    The wrapper registers start_game, submit_response, and get_state as tools on
    its own FastMCP instance while delegating all game logic to the wrapped
    environment. It also keeps the OpenEnv environment lifecycle intact, so the
    same object can be driven either through MCP tool calls or through the
    regular reset and step interface.
    """

    def __init__(self, clem_env: SelectableClemGameEnvironment, completion_path: str | Path | None = None):

        self.clem_env = clem_env
        self._completion_path = (Path(completion_path) if completion_path is not None else None)

        mcp = FastMCP("clem-game")

        super().__init__(mcp)

        @self.tool()
        def start_game(game_id: int | None = None, experiment_name: str | None = None) -> dict:
            """start/reset the current clembench game and return the initial observation."""
            reset_kwargs = {}

            if game_id is not None:
                reset_kwargs["game_id"] = game_id

            if experiment_name is not None:
                reset_kwargs["experiment_name"] = experiment_name

            return self._record_terminal_observation(self.clem_env.reset(**reset_kwargs))

        @self.tool()
        def submit_response(response: str) -> dict:
            """submit a response/move to the current clembench game."""
            return self._record_terminal_observation(self.clem_env.step(ClemGameAction(response=response)))

        @self.tool()
        def get_state() -> dict:
            """get the current clembench game state."""
            return self.clem_env.state.model_dump()

        @self.tool()
        def abort_game(reason: str) -> dict:
            """abort the active game after an external-agent control failure."""
            return self._record_terminal_observation(self.clem_env.abort(reason))

    def _record_terminal_observation(self, observation: ClemGameObservation) -> dict:
        """Return an MCP observation and atomically mark a completed episode."""
        result = _observation_to_dict(observation)

        if result["done"] is True and self._completion_path is not None:
            payload = {"done": True, "reward": result["reward"], "metadata": result["metadata"], "source": "host_mcp"}
            temporary_path = self._completion_path.with_suffix(self._completion_path.suffix + ".tmp")
            self._completion_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
            os.replace(temporary_path, self._completion_path)

        return result

    def reset(self, seed=None, episode_id=None, **kwargs) -> ClemGameObservation:
        observation = self.clem_env.reset(seed=seed, episode_id=episode_id, **kwargs)
        self._record_terminal_observation(observation)
        return observation

    def _step_impl(self, action: Action, timeout_s: float | None = None, **kwargs) -> Observation:
        if isinstance(action, ClemGameAction):
            observation = self.clem_env.step(action, timeout_s=timeout_s, **kwargs)
            self._record_terminal_observation(observation)
            return observation

        raise TypeError(f"Unsupported action type: {type(action)}")

    @property
    def state(self) -> ClemGameState:
        return self.clem_env.state

    def close(self) -> None:
        self.clem_env.close()
        super().close()
