import base64
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

from clemagents.mcp.environment import (
    SelectableClemGameEnvironment,
    _observation_to_dict,
)
from clemcore.clemgame.envs.openenv.models import ClemGameAction
from clemcore.clemgame.master import GameState, Outcome
from clemcore.clemgame.envs.pettingzoo.master import GameMasterEnv
from clemcore.clemgame.envs.pettingzoo.wrappers import SinglePlayerWrapper
from clemcore.clemgame.registry import GameSpec


class TestExternalAgentMCPEnvironment(unittest.TestCase):
    def test_full_environment_selects_instance_and_finalizes_callbacks(self):
        with tempfile.TemporaryDirectory() as directory:
            game_path = Path(directory)
            (game_path / "in").mkdir()
            (game_path / "in" / "instances.json").write_text(json.dumps({
                "experiments": [{"name": "fixture", "game_instances": [
                    {"game_id": 1}, {"game_id": 2},
                ]}],
            }))
            spec = GameSpec(game_name="fixture", game_path=str(game_path), players=1)
            player = SimpleNamespace(name="Player", model=SimpleNamespace(name="fixture"))
            master = MagicMock()
            master.state = GameState()
            master.current_player = player
            master.get_players.return_value = [player]
            master.get_context_for.return_value = {"role": "user", "content": "Initial clue"}
            benchmark = MagicMock(game_spec=spec)
            benchmark.create_game_master.return_value = master
            callbacks = MagicMock()

            def finish(response):
                self.assertEqual(response, "answer")
                master.state.succeed()
                master.get_context_for.return_value = {"role": "user", "content": "Solved"}
                return True, {}

            master.step.side_effect = finish
            with patch("clemagents.mcp.environment.GameRegistry") as registry, \
                 patch("clemcore.clemgame.benchmark.GameBenchmark.load_from_spec", return_value=benchmark):
                registry.from_directories_and_cwd_files.return_value.get_game_spec.return_value = spec
                environment = SelectableClemGameEnvironment("fixture", callbacks=callbacks)
                try:
                    initial = environment.reset(experiment_name="fixture", game_id=2)
                    self.assertFalse(initial.done)
                    master.setup.assert_called_once_with(game_id=2)
                    final = environment.step(ClemGameAction(response="answer"))
                    self.assertTrue(final.done)
                    self.assertEqual(final.context["content"], "Solved")
                    self.assertEqual(final.reward, 1.0)
                    self.assertEqual(environment._game_env.agents, [])
                    callbacks.on_game_start.assert_called_once()
                    callbacks.on_game_step.assert_called_once()
                    callbacks.on_game_end.assert_called_once()
                finally:
                    environment.close()
                callbacks.on_benchmark_end.assert_called_once()

    def _environment(self,
                     displayed_context,
                     stored_context,
                     returned_context,
                     outcome,
                     error=None):
        player = SimpleNamespace(name="Player 1")
        game_master = SimpleNamespace(
            current_player=player,
            state=SimpleNamespace(outcome=outcome, error=error),
            context_for_player={player.name: stored_context},
            get_context_for=lambda selected_player: displayed_context,
        )
        game_environment = MagicMock()
        game_environment.unwrapped.game_master = game_master
        game_environment.agent_selection = "player_0"
        game_environment.last.return_value = (
            returned_context,
            -1.0 if outcome == Outcome.ABORTED else 1.0,
            True,
            False,
            {},
        )
        environment = SelectableClemGameEnvironment.__new__(
            SelectableClemGameEnvironment
        )
        environment._game_env = game_environment
        environment._learner_agent = "player_0"
        environment._state = SimpleNamespace(step_count=0)
        return environment

    def test_stale_terminal_context_is_replaced_with_outcome_and_reason(self):
        displayed_context = {
            "role": "user",
            "content": "Initial prompt\n\nInitial game message",
        }
        stored_context = {
            "role": "user",
            "content": "Initial game message",
        }
        error = SimpleNamespace(reason="The response format was invalid.")
        environment = self._environment(
            displayed_context,
            stored_context,
            stored_context,
            Outcome.ABORTED,
            error,
        )

        observation = environment.step(ClemGameAction(response="bad response"))

        self.assertEqual(
            observation.context,
            {
                "role": "user",
                "content": (
                    "The game ended before normal completion.\n"
                    "Reason: The response format was invalid."
                ),
            },
        )
        self.assertEqual(observation.metadata["outcome"], "aborted")
        self.assertEqual(
            observation.metadata["terminal_reason"],
            "The response format was invalid.",
        )
        self.assertTrue(observation.metadata["terminal_context_normalized"])

    def test_unsuccessful_normal_completion_is_not_called_a_failure(self):
        context = {"role": "user", "content": "Final clue"}
        environment = self._environment(
            context,
            context,
            context,
            Outcome.FAILURE,
        )

        observation = environment.step(ClemGameAction(response="wrong answer"))

        self.assertEqual(
            observation.context,
            {
                "role": "user",
                "content": "The game is complete. The objective was not achieved.",
            },
        )
        self.assertEqual(observation.metadata["outcome"], "failure")
        self.assertTrue(observation.metadata["terminal_context_normalized"])

    def test_genuine_terminal_context_is_preserved(self):
        initial_context = {"role": "user", "content": "Initial game message"}
        final_context = {"role": "user", "content": "You solved the game."}
        environment = self._environment(
            initial_context,
            initial_context,
            final_context,
            Outcome.SUCCESS,
        )

        observation = environment.step(ClemGameAction(response="final response"))

        self.assertEqual(observation.context, final_context)
        self.assertEqual(observation.metadata["outcome"], "success")
        self.assertNotIn("terminal_reason", observation.metadata)
        self.assertNotIn("terminal_context_normalized", observation.metadata)

    def test_running_turn_returns_feedback_without_cleanup(self):
        initial = {"role": "user", "content": "Initial clue"}
        feedback = {"role": "user", "content": "Try again", "image": ["next.png"]}
        environment = self._environment(initial, initial, feedback, Outcome.RUNNING)
        env = environment._game_env
        env.last.return_value = (feedback, 0.5, False, False, {"turn": 2})

        observation = environment.step(ClemGameAction(response="guess"))

        self.assertFalse(observation.done)
        self.assertEqual(observation.context, feedback)
        self.assertEqual(observation.reward, 0.5)
        self.assertEqual(observation.metadata["turn"], 2)
        self.assertEqual(environment._state.step_count, 1)
        env.step.assert_called_once_with("guess")

    def test_autoplay_already_removed_all_agents(self):
        context = {"role": "user", "content": "Final context"}
        environment = self._environment(context, context, context, Outcome.SUCCESS)
        env = environment._game_env
        env.agent_selection = None
        env.observe.return_value = context
        # PettingZoo removes the per-agent dictionaries during cleanup.
        env.rewards = env.terminations = env.truncations = env.infos = {}

        observation = environment.step(ClemGameAction(response="answer"))

        self.assertTrue(observation.done)
        self.assertEqual(observation.metadata["outcome"], "success")
        env.last.assert_not_called()
        env.observe.assert_called_once_with("player_0")
        env.step.assert_called_once_with("answer")

    def test_reset_can_end_before_first_learner_turn(self):
        context = {"role": "user", "content": "The game ended before your turn."}
        environment = self._environment(context, context, context, Outcome.ABORTED)
        env = environment._game_env
        env.agent_selection = None
        env.observe.return_value = context
        env.rewards = env.terminations = env.truncations = env.infos = {}
        environment._game_name = "test"
        environment._game_spec = MagicMock()
        environment._game_instance_split = None
        environment._single_pass = True
        environment._callbacks = environment._reward_func = environment._feedback_func = None
        environment._env_agents = {}
        environment._state = SimpleNamespace(game_name="test", step_count=0, episode_count=0)
        module = "clemagents.mcp.environment"
        with patch(module + ".GameBenchmarkWrapper"), patch(module + ".OrderEnforcingWrapper"), \
             patch(module + ".GameInstances") as instances, \
             patch(module + ".GameInstanceIteratorWrapper"), \
             patch(module + ".SinglePlayerWrapper", return_value=env):
            instances.from_game_spec.return_value.filter.return_value = [object()]
            observation = environment.reset(experiment_name="test", game_id=1)

        self.assertTrue(observation.done)
        self.assertEqual(observation.context, context)
        self.assertEqual(environment._state.episode_count, 1)
        env.last.assert_not_called()

    def test_real_environment_autoplay_preserves_rewards_and_callbacks(self):
        for player_count in (1, 2):
            with self.subTest(player_count=player_count):
                players = [SimpleNamespace(name=f"Player {i}", model=SimpleNamespace(name="test"))
                           for i in range(player_count)]
                master = MagicMock()
                master.state = GameState()
                master.current_player = players[0]
                master.get_context_for.return_value = {"role": "user", "content": "A clue"}
                callbacks = MagicMock()
                env = GameMasterEnv(MagicMock(), callbacks=callbacks)
                env.game_master = master
                env.game_instance = {}
                env.agents = [f"player_{i}" for i in range(player_count)]
                env.possible_agents = env.agents.copy()
                env.player_by_agent_id = dict(zip(env.agents, players))
                env.player_to_agent_id = {p.name: a for a, p in env.player_by_agent_id.items()}
                env.agent_selection = "player_0"
                env.terminations = dict.fromkeys(env.agents, False)
                env.truncations = dict.fromkeys(env.agents, False)
                env.rewards = dict.fromkeys(env.agents, 0.0)
                env._cumulative_rewards = env.rewards.copy()
                env.infos = {a: {} for a in env.agents}

                def game_step(response):
                    if response == "answer" and player_count == 2:
                        master.current_player = players[1]
                        return False, {}
                    master.state.succeed()
                    return True, {"final": True}

                master.step.side_effect = game_step
                native_players = {"player_1": lambda context: "automatic answer"} if player_count == 2 else {}
                environment = SelectableClemGameEnvironment.__new__(SelectableClemGameEnvironment)
                environment._game_env = SinglePlayerWrapper(env, "player_0", native_players)
                environment._learner_agent = "player_0"
                environment._state = SimpleNamespace(step_count=0)

                observation = environment.step(ClemGameAction(response="answer"))

                self.assertTrue(observation.done)
                self.assertEqual(observation.reward, 1.0)
                self.assertEqual(observation.metadata["outcome"], "success")
                self.assertEqual(env.agents, [])
                self.assertIsNone(env.agent_selection)
                callbacks.on_game_end.assert_called_once()
                self.assertEqual(callbacks.on_game_step.call_count, player_count)
                self.assertEqual(master.step.call_count, player_count)

    def test_truncation_reads_result_before_cleanup(self):
        context = {"role": "user", "content": "Last clue"}
        environment = self._environment(context, context, context, Outcome.RUNNING)
        env = environment._game_env
        env.last.return_value = (context, 0.25, False, True, {"limit": "turns"})

        observation = environment.step(ClemGameAction(response="answer"))

        self.assertEqual(observation.reward, 0.25)
        self.assertTrue(observation.metadata["truncated"])
        self.assertEqual(observation.metadata["outcome"], "truncated")
        self.assertEqual(env.step.call_args_list, [call("answer"), call(None)])

    def test_control_abort_uses_normal_step_and_cleanup(self):
        context = {"role": "user", "content": "Last clue"}
        environment = self._environment(context, context, context, Outcome.ABORTED)
        environment._game_env.unwrapped.game_master.state = GameState()

        observation = environment.abort("timeout")

        self.assertTrue(observation.done)
        self.assertTrue(observation.metadata["control_failure"])
        self.assertEqual(observation.reward, -1.0)
        self.assertEqual(environment._game_env.step.call_args_list, [call("timeout"), call(None)])

    def test_observation_images_are_encoded_before_container_transport(self):
        image_bytes = b"not-a-real-png-but-sufficient-for-transport"

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "location.png"
            image_path.write_bytes(image_bytes)
            observation = SimpleNamespace(
                context={
                    "role": "user",
                    "content": "Identify this location.",
                    "image": [str(image_path)],
                },
                reward=None,
                done=False,
                metadata={},
            )
            result = _observation_to_dict(observation)

        self.assertEqual(result["context"]["content"], "Identify this location.")
        self.assertEqual(
            result["context"]["image"],
            [{
                "data": base64.b64encode(image_bytes).decode("ascii"),
                "mimeType": "image/png",
            }],
        )


if __name__ == "__main__":
    unittest.main()
