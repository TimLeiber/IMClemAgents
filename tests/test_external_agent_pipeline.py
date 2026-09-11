from clemagents.adapters.traces.codex import _codex_tool_result_text
import asyncio
import json
import logging
import os
import sys
import subprocess
import tempfile
import time
import unittest
from datetime import datetime, timezone
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from unittest.mock import MagicMock, patch

import requests

from clemagents.adapters import model_connection
from clemagents.adapters import harness_class_for_agent
from clemagents.adapters.claude_code import (
    ClaudeCodeHarness,
    _anthropic_proxy_base_url,
)
from clemagents.adapters.codex import CodexHarness
from clemagents.adapters.hermes import HermesHarness
from clemagents.adapters.openai_compatible_proxy import (
    OpenAICompatibleProxy,
    UPSTREAM_REQUEST_TIMEOUT,
    _normalize_response_output_order,
    _prepare_request_body,
)
from clemagents.adapters.openclaw import (
    OpenClawHarness,
    _openclaw_gateway,
    _validate_openclaw_model_connection,
)
from clemagents.mcp.bridge import (
    CONTROL_FAILURE_RESPONSE,
    OpenEnvMCPClient,
    REPEATED_START_MESSAGE,
    create_mcp_bridge,
)
from clemagents.mcp.server import (
    _SuppressFastMCPToolCallErrors,
    result_run_dir_name,
)
from clemagents.run_pipeline.utils import (
    _stop_docker_container,
    run_docker_episode,
    write_agent_artifacts,
)
from clemagents.adapters.utils import (
    mcp_environment,
    run_process_until_game_complete,
)
from clemagents.transcribe_agent_loop import build_agent_loop_html


class TestExternalAgentPipeline(unittest.TestCase):
    def test_mcp_environment_forwards_generic_observation_directory(self):
        with patch.dict(
            "os.environ",
            {"GAME_OBSERVATION_DIR": "/workspace/game_observations"},
        ):
            environment = mcp_environment("http://host.example/mcp")

        self.assertEqual(
            environment["GAME_OBSERVATION_DIR"],
            "/workspace/game_observations",
        )

    def test_fastmcp_tool_call_errors_are_hidden_from_host_log(self):
        log_filter = _SuppressFastMCPToolCallErrors()
        tool_error = logging.LogRecord(
            "fastmcp.server.server",
            logging.ERROR,
            "",
            0,
            "Error calling tool %r",
            ("submit_response",),
            None,
        )
        server_error = logging.LogRecord(
            "fastmcp.server.server",
            logging.ERROR,
            "",
            0,
            "MCP server failed",
            (),
            None,
        )

        self.assertFalse(log_filter.filter(tool_error))
        self.assertTrue(log_filter.filter(server_error))

    def test_openai_compatible_proxy_has_no_inference_read_timeout(self):
        self.assertEqual(UPSTREAM_REQUEST_TIMEOUT, (30, None))

    def test_docker_cleanup_contains_unresponsive_docker_commands(self):
        class AttachedProcess:
            def __init__(self):
                self.terminated = False
                self.killed = False

            def poll(self):
                return None

            def terminate(self):
                self.terminated = True

            def wait(self, timeout=None):
                raise subprocess.TimeoutExpired(["docker", "run"], timeout)

            def kill(self):
                self.killed = True

        with tempfile.TemporaryDirectory() as directory:
            cidfile_path = Path(directory) / "container.cid"
            cidfile_path.write_text("container-id", encoding="utf-8")
            process = AttachedProcess()

            with patch(
                "clemagents.run_pipeline.utils.subprocess.run",
                side_effect=subprocess.TimeoutExpired(["docker"], 10),
            ) as docker_command:
                _stop_docker_container(process, cidfile_path)

        self.assertEqual(docker_command.call_count, 2)
        self.assertTrue(process.terminated)
        self.assertTrue(process.killed)

    def test_docker_output_replaces_invalid_utf8(self):
        with tempfile.TemporaryDirectory() as directory:
            keys_path = Path(directory) / "key.json"
            registry_path = Path(directory) / "agent_registry.json"
            keys_path.write_text("{}", encoding="utf-8")
            registry_path.write_text("[]", encoding="utf-8")
            actual_popen = subprocess.Popen
            script = (
                "import sys; "
                "sys.stdout.buffer.write(b'valid\\n\\xe2broken\\n'); "
                "sys.stdout.buffer.flush()"
            )

            def simulated_container(_command, **kwargs):
                return actual_popen([sys.executable, "-c", script], **kwargs)

            with patch(
                "clemagents.run_pipeline.utils.KEYS_PATH",
                keys_path,
            ), patch(
                "clemagents.run_pipeline.utils.REGISTRY_PATH",
                registry_path,
            ), patch(
                "clemagents.run_pipeline.utils.subprocess.Popen",
                side_effect=simulated_container,
            ):
                trace, returncode, _ = run_docker_episode(
                    experiment_name="test",
                    game_id=1,
                    agent_name="test-agent",
                    episode_timeout=5,
                )

        self.assertEqual(returncode, 0)
        self.assertIn("valid", trace)
        self.assertIn("\ufffdbroken", trace)

    def test_responses_proxy_preserves_streamed_reasoning_tool_order(self):
        events = [
            {
                "type": "response.output_item.added",
                "sequence_number": 2,
                "output_index": 0,
                "item": {
                    "id": "reasoning-1",
                    "type": "reasoning",
                    "summary": [],
                },
            },
            {
                "type": "response.reasoning_summary_text.delta",
                "sequence_number": 3,
                "output_index": 0,
                "item_id": "reasoning-1",
                "delta": "I should start the game.",
            },
            {
                "type": "response.output_item.added",
                "sequence_number": 4,
                "output_index": 0,
                "item": {
                    "id": "function-1",
                    "type": "function_call",
                    "name": "start_game",
                    "call_id": "call-1",
                    "arguments": "",
                },
            },
            {
                "type": "response.output_item.done",
                "sequence_number": 5,
                "output_index": 0,
                "item": {
                    "id": "function-1",
                    "type": "function_call",
                    "name": "start_game",
                    "call_id": "call-1",
                    "arguments": "{}",
                },
            },
            {
                "type": "response.output_item.done",
                "sequence_number": 6,
                "output_index": 1,
                "item": {
                    "id": "reasoning-1",
                    "type": "reasoning",
                    "summary": [{
                        "type": "summary_text",
                        "text": "I should start the game.",
                    }],
                },
            },
            {
                "type": "response.completed",
                "sequence_number": 7,
                "response": {
                    "output": [
                        {
                            "id": "function-1",
                            "type": "function_call",
                            "name": "start_game",
                            "call_id": "call-1",
                            "arguments": "{}",
                        },
                        {
                            "id": "reasoning-1",
                            "type": "reasoning",
                            "summary": [{
                                "type": "summary_text",
                                "text": "I should start the game.",
                            }],
                        },
                    ],
                },
            },
        ]
        body = "".join(
            f"data: {json.dumps(event)}\n\n"
            for event in events
        ).encode("utf-8")

        ordered = _normalize_response_output_order(body)
        rewritten = ordered.decode("utf-8")
        rewritten_events = [
            json.loads(line[6:])
            for line in rewritten.splitlines()
            if line.startswith("data: ")
        ]
        added_events = [
            event
            for event in rewritten_events
            if event["type"] == "response.output_item.added"
        ]
        completed = next(
            event
            for event in rewritten_events
            if event["type"] == "response.completed"
        )

        self.assertEqual(
            [event["output_index"] for event in added_events],
            [0, 1],
        )
        self.assertEqual(
            [item["type"] for item in completed["response"]["output"]],
            ["reasoning", "function_call"],
        )
        self.assertEqual(
            completed["response"]["output"][1]["name"],
            "start_game",
        )

    def test_responses_proxy_respects_valid_provider_output_indexes(self):
        events = [
            {
                "type": "response.output_item.added",
                "sequence_number": 2,
                "output_index": 1,
                "item": {
                    "id": "function-1",
                    "type": "function_call",
                    "name": "start_game",
                },
            },
            {
                "type": "response.output_item.added",
                "sequence_number": 3,
                "output_index": 0,
                "item": {
                    "id": "reasoning-1",
                    "type": "reasoning",
                    "summary": [],
                },
            },
            {
                "type": "response.completed",
                "sequence_number": 4,
                "response": {
                    "output": [
                        {
                            "id": "function-1",
                            "type": "function_call",
                            "name": "start_game",
                        },
                        {
                            "id": "reasoning-1",
                            "type": "reasoning",
                            "summary": [],
                        },
                    ],
                },
            },
        ]
        body = "".join(
            f"data: {json.dumps(event)}\n\n"
            for event in events
        ).encode("utf-8")

        ordered = _normalize_response_output_order(body)
        rewritten = ordered.decode("utf-8")
        completed = next(
            json.loads(line[6:])
            for line in rewritten.splitlines()
            if line.startswith("data: ")
            and json.loads(line[6:])["type"] == "response.completed"
        )

        self.assertEqual(
            [item["type"] for item in completed["response"]["output"]],
            ["reasoning", "function_call"],
        )

    def test_codex_tool_result_renders_only_standardized_game_message(self):
        output = (
            "Wall time: 0.1 seconds\n"
            "Process exited with code 0\n"
            "Final output:\n"
            "Output:\n"
            + json.dumps({"result": "Initial game message"})
        )

        self.assertEqual(
            _codex_tool_result_text(output),
            "Initial game message"
        )

    def test_harness_class_is_resolved_from_agent_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            registry_path = Path(directory) / "agent_registry.json"
            registry_path.write_text(
                json.dumps([{
                    "agent_name": "codex-test",
                    "backend": "codex",
                    "agent_config": {}
                }]),
                encoding="utf-8"
            )

            harness_class = harness_class_for_agent("codex-test", registry_path)

        self.assertIs(harness_class, CodexHarness)

    def test_final_agent_trace_is_serialized_after_persistence(self):
        trace = "\n".join([
            "agent_loop_instruction_start",
            "Play through MCP tools.",
            "agent_loop_instruction_end",
            "raw_upstream_request_1_start",
            "path: /v1/responses",
            json.dumps({"instructions": "Native instruction", "input": []}),
            "raw_upstream_request_1_end",
            "raw_upstream_response_2_start",
            "path: /v1/responses",
            "status: 200",
            "content_type: application/json",
            "content_encoding: ",
            json.dumps({"output": []}),
            "raw_upstream_response_2_end"
        ])

        with tempfile.TemporaryDirectory() as directory:
            results_dir = Path(directory) / "results"
            instance_dir = (
                results_dir
                / "codex-test"
                / "wordle"
                / "medium"
                / "instance_00001"
            )
            instance_dir.mkdir(parents=True)
            (instance_dir / "instance.json").write_text(
                json.dumps({"game_id": 1}),
                encoding="utf-8"
            )
            registry_path = Path(directory) / "agent_registry.json"
            registry_path.write_text(
                json.dumps([{
                    "agent_name": "codex-test",
                    "backend": "codex",
                    "agent_config": {}
                }]),
                encoding="utf-8"
            )

            trace_path = write_agent_artifacts(
                results_dir=str(results_dir),
                run_dir="codex-test",
                game_name="wordle",
                experiment_name="medium",
                game_id=1,
                before_instance_dirs=set(),
                trace_text=trace,
                metadata={
                    "agent": "codex-test",
                    "started_at": datetime.now(timezone.utc).isoformat()
                },
                registry_path=registry_path
            )
            agent_loop = json.loads(
                (instance_dir / "agent_loop.json").read_text(encoding="utf-8")
            )

        self.assertEqual(trace_path, instance_dir / "agent_trace.log")
        self.assertEqual(agent_loop["capture"]["model_requests"]["count"], 1)
        self.assertEqual(agent_loop["capture"]["model_responses"]["count"], 1)
        self.assertGreater(len(agent_loop["events"]), 1)

    def test_finalized_native_artifacts_are_preserved_without_overwriting_game_files(self):
        with tempfile.TemporaryDirectory() as directory:
            results_dir = Path(directory) / "results"
            instance_dir = (
                results_dir
                / "test-agent"
                / "wordle"
                / "medium"
                / "instance_00001"
            )
            instance_dir.mkdir(parents=True)
            (instance_dir / "instance.json").write_text(
                json.dumps({"game_id": 1}),
                encoding="utf-8",
            )
            (instance_dir / "interactions.json").write_text(
                "game-owned",
                encoding="utf-8",
            )
            source_dir = Path(directory) / "finalized-artifacts"
            source_dir.mkdir()
            (source_dir / "native_session.jsonl").write_text(
                "native artifact",
                encoding="utf-8",
            )
            (source_dir / "interactions.json").write_text(
                "must not overwrite",
                encoding="utf-8",
            )

            write_agent_artifacts(
                results_dir=str(results_dir),
                run_dir="test-agent",
                game_name="wordle",
                experiment_name="medium",
                game_id=1,
                before_instance_dirs=set(),
                trace_text="host trace",
                metadata={
                    "started_at": datetime.now(timezone.utc).isoformat(),
                },
                source_artifact_dir=source_dir,
            )

            self.assertEqual(
                (
                    instance_dir
                    / "agent_artifacts"
                    / "finalized-artifacts"
                    / "native_session.jsonl"
                ).read_text(encoding="utf-8"),
                "native artifact",
            )
            self.assertEqual(
                (instance_dir / "interactions.json").read_text(encoding="utf-8"),
                "game-owned",
            )
            self.assertEqual(
                (instance_dir / "agent_trace.log").read_text(encoding="utf-8"),
                "host trace",
            )

    def test_finalized_native_artifacts_preserve_container_only_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            results_dir = Path(directory) / "results"
            instance_dir = (
                results_dir
                / "test-agent"
                / "geolocate"
                / "civic_public"
                / "instance_00003"
            )
            instance_dir.mkdir(parents=True)
            (instance_dir / "instance.json").write_text(
                json.dumps({"game_id": 3}),
                encoding="utf-8",
            )
            source_dir = Path(directory) / "finalized-artifacts"
            source_dir.mkdir()
            source_link = source_dir / "browser-automation"
            source_link.symlink_to(
                "/container-only/plugin-skills/browser-automation",
                target_is_directory=True,
            )

            write_agent_artifacts(
                results_dir=str(results_dir),
                run_dir="test-agent",
                game_name="geolocate",
                experiment_name="civic_public",
                game_id=3,
                before_instance_dirs=set(),
                trace_text="host trace",
                metadata={
                    "started_at": datetime.now(timezone.utc).isoformat(),
                },
                source_artifact_dir=source_dir,
            )

            copied_link = (
                instance_dir
                / "agent_artifacts"
                / "finalized-artifacts"
                / "browser-automation"
            )
            self.assertTrue(copied_link.is_symlink())
            self.assertEqual(copied_link.readlink(), source_link.readlink())

    def test_codex_trace_orders_reasoning_before_terminal_tool_call(self):
        response_events = [
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "id": "call-item-1",
                    "call_id": "call-1",
                    "name": "mcp__clem_game__start_game",
                    "arguments": "{}"
                }
            },
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "reasoning",
                    "id": "reasoning-1",
                    "content": [{
                        "type": "reasoning_text",
                        "text": "I should start the game."
                    }]
                }
            }
        ]
        trace = "\n".join([
            "raw_upstream_request_1_start",
            "path: /v1/responses",
            json.dumps({"instructions": "Native instruction", "input": []}),
            "raw_upstream_request_1_end",
            "raw_upstream_response_2_start",
            "path: /v1/responses",
            "status: 200",
            "content_type: text/event-stream",
            "content_encoding: ",
            *[f"data: {json.dumps(event)}" for event in response_events],
            "raw_upstream_response_2_end"
        ])

        with tempfile.TemporaryDirectory() as directory:
            episode_dir = Path(directory)
            (episode_dir / "agent_trace.log").write_text(trace, encoding="utf-8")
            parsed = CodexHarness.parse_agent_trace(episode_dir)

        semantic_events = [event for event in parsed["events"]
                           if event["type"] in {"reasoning", "tool_call"}]
        self.assertEqual(
            [event["type"] for event in semantic_events],
            ["reasoning", "tool_call"]
        )
        self.assertEqual(semantic_events[0]["content"], "I should start the game.")
        self.assertEqual(
            semantic_events[1]["name"],
            "mcp__clem_game__start_game"
        )

    def test_codex_trace_keeps_failed_provider_tool_attempts(self):
        def failed_search_response(response_id, item_id, query, reasoning=None):
            response_events = []

            if reasoning is not None:
                response_events.append({
                    "type": "response.output_item.done",
                    "item": {
                        "type": "reasoning",
                        "id": f"reasoning-{response_id}",
                        "summary": [{
                            "type": "summary_text",
                            "text": reasoning
                        }]
                    }
                })

            response_events.extend([
                {
                    "type": "response.output_item.added",
                    "item": {
                        "type": "web_search_call",
                        "id": item_id,
                        "status": "in_progress",
                        "action": {"type": "search", "query": query}
                    }
                },
                {
                    "type": "response.failed",
                    "response": {
                        "id": response_id,
                        "status": "failed",
                        "output": [],
                        "error": {
                            "code": "authentication_error",
                            "message": "web search authentication failed"
                        }
                    }
                }
            ])
            return response_events

        trace_lines = []

        for turn, events in enumerate([
            failed_search_response(
                "response-1",
                "search-1",
                "first query",
                reasoning="I should search."
            ),
            failed_search_response(
                "response-2",
                "search-2",
                "second query"
            )
        ], start=1):
            trace_lines.extend([
                f"raw_upstream_request_{turn}_start",
                "path: /v1/responses",
                json.dumps({"instructions": "Native instruction", "input": []}),
                f"raw_upstream_request_{turn}_end",
                f"raw_upstream_response_{turn}_start",
                "path: /v1/responses",
                "status: 200",
                "content_type: text/event-stream",
                "content_encoding: ",
                *[f"data: {json.dumps(event)}" for event in events],
                f"raw_upstream_response_{turn}_end"
            ])

        trace_lines.extend([
            "codex_stdout_jsonl:",
            json.dumps({
                "type": "turn.failed",
                "error": {"message": "stream disconnected before completion"}
            })
        ])

        with tempfile.TemporaryDirectory() as directory:
            episode_dir = Path(directory)
            (episode_dir / "agent_trace.log").write_text(
                "\n".join(trace_lines),
                encoding="utf-8"
            )
            parsed = CodexHarness.parse_agent_trace(episode_dir)

        visible_events = [
            event
            for event in parsed["events"]
            if event["type"] in {"reasoning", "tool_call", "tool_result", "error"}
        ]
        self.assertEqual(
            [event["type"] for event in visible_events],
            [
                "reasoning",
                "tool_call",
                "tool_result",
                "tool_call",
                "tool_result",
                "error"
            ]
        )
        self.assertEqual(
            [event["arguments"] for event in visible_events
             if event["type"] == "tool_call"],
            [{"query": "first query"}, {"query": "second query"}]
        )
        self.assertTrue(all(
            event["status"] == "failed"
            for event in visible_events
            if event["type"] in {"tool_call", "tool_result", "error"}
        ))
        self.assertTrue(all(
            "authentication_error" in event["content"]
            for event in visible_events
            if event["type"] == "tool_result"
        ))
        self.assertEqual(visible_events[-1]["turn"], 2)

    def test_claude_proxy_base_does_not_duplicate_v1(self):
        self.assertEqual(
            _anthropic_proxy_base_url("http://127.0.0.1:1234/api/v1"),
            "http://127.0.0.1:1234/api",
        )

    def test_claude_code_trace_parser_standardizes_sdk_messages(self):
        trace = "\n".join([
            "agent_loop_instruction_start",
            "Play through MCP tools.",
            "agent_loop_instruction_end",
            "SystemMessage(subtype='init', data={'tools': ['WebSearch', 'mcp__clem_game__start_game'], 'model': 'test-model', 'permissionMode': 'bypassPermissions'})",
            "SystemMessage(subtype='thinking_tokens', data={'estimated_tokens': 4, 'estimated_tokens_delta': 4})",
            "AssistantMessage(content=[ThinkingBlock(thinking='I should search.', signature='')], message_id='message-1')",
            "AssistantMessage(content=[TextBlock(text='I will verify this.')], message_id='message-1')",
            "AssistantMessage(content=[ToolUseBlock(id='call-1', name='WebSearch', input={'query': 'example'})], message_id='message-1')",
            "UserMessage(content=[ToolResultBlock(tool_use_id='call-1', content=" + repr(json.dumps({
                "context": {"role": "user", "content": "Game prompt"},
                "reward": None,
                "done": False,
                "metadata": {"truncated": False}
            })) + ", is_error=None)])",
            "AssistantMessage(content=[TextBlock(text='DONE')], message_id='message-2')",
            "ResultMessage(subtype='success', duration_ms=20, is_error=False, num_turns=2, result='DONE')",
            "success: True",
        ])

        with tempfile.TemporaryDirectory() as directory:
            episode_dir = Path(directory)
            (episode_dir / "agent_trace.log").write_text(trace, encoding="utf-8")
            parsed = ClaudeCodeHarness.parse_agent_trace(episode_dir)

        event_types = [event["type"] for event in parsed["events"]]
        self.assertEqual(parsed["backend"], "claude_code")
        self.assertIn("instruction", event_types)
        self.assertIn("reasoning", event_types)
        self.assertIn("tool_call", event_types)
        self.assertIn("tool_result", event_types)
        self.assertIn("tool_preamble", event_types)
        self.assertIn("assistant_text", event_types)
        self.assertEqual(parsed["runtime"]["model"], "test-model")
        self.assertEqual(parsed["result"]["num_turns"], 2)
        self.assertEqual(parsed["capture"]["thinking_tokens"]["estimated_total"], 4)
        tool_result = next(event for event in parsed["events"] if event["type"] == "tool_result")
        self.assertEqual(tool_result["content"], "Game prompt")
        self.assertFalse(tool_result["done"])
        self.assertEqual(tool_result["metadata"], {"truncated": False})

    def test_claude_code_trace_parser_flattens_text_tool_result_blocks(self):
        trace = "\n".join([
            "AssistantMessage(content=[ToolUseBlock(id='call-1', name='mcp__clem_game__start_game', input={})], message_id='message-1')",
            "UserMessage(content=[ToolResultBlock(tool_use_id='call-1', content=[{'type': 'text', 'text': 'Initial game message'}], is_error=None)])",
        ])

        with tempfile.TemporaryDirectory() as directory:
            episode_dir = Path(directory)
            (episode_dir / "agent_trace.log").write_text(trace, encoding="utf-8")
            parsed = ClaudeCodeHarness.parse_agent_trace(episode_dir)

        tool_result = next(
            event for event in parsed["events"] if event["type"] == "tool_result"
        )
        self.assertEqual(tool_result["content"], "Initial game message")
        self.assertIsInstance(tool_result["payload"]["content"], list)

    def test_claude_code_trace_parser_preserves_subagent_timeline(self):
        trace = "\n".join([
            "AssistantMessage(content=[ToolUseBlock(id='agent-call', name='Agent', input={'description': 'Solve Wordle'})], message_id='parent-message')",
            "TaskStartedMessage(subtype='task_started', data={'task_id': 'agent-1', 'tool_use_id': 'agent-call', 'description': 'Solve Wordle', 'subagent_type': 'general-purpose', 'task_type': 'local_agent', 'prompt': 'Find the next guess.'}, task_id='agent-1', tool_use_id='agent-call')",
            "AssistantMessage(content=[ThinkingBlock(thinking='I should test letters.', signature='')], message_id='child-message', parent_tool_use_id='agent-call')",
            "AssistantMessage(content=[ToolUseBlock(id='child-call', name='mcp__game__submit_response', input={'response': 'guess: crane'})], message_id='child-message', parent_tool_use_id='agent-call')",
            "TaskProgressMessage(subtype='task_progress', data={'task_id': 'agent-1', 'tool_use_id': 'agent-call', 'description': 'Solve Wordle', 'subagent_type': 'general-purpose', 'usage': {'total_tokens': 120, 'tool_uses': 1}, 'last_tool_name': 'mcp__game__submit_response'}, task_id='agent-1', tool_use_id='agent-call')",
            "UserMessage(content=[ToolResultBlock(tool_use_id='child-call', content='feedback', is_error=None)], parent_tool_use_id='agent-call')",
            "TaskNotificationMessage(subtype='task_notification', data={'task_id': 'agent-1', 'tool_use_id': 'agent-call', 'status': 'completed', 'summary': 'Solved Wordle', 'output_file': '/tmp/agent.out'}, task_id='agent-1', tool_use_id='agent-call', status='completed', summary='Solved Wordle', output_file='/tmp/agent.out')",
            "ResultMessage(subtype='success', duration_ms=20, is_error=False, num_turns=2, result='DONE')",
        ])

        with tempfile.TemporaryDirectory() as directory:
            episode_dir = Path(directory)
            (episode_dir / "agent_trace.log").write_text(trace, encoding="utf-8")
            parsed = ClaudeCodeHarness.parse_agent_trace(episode_dir)

        event_kinds = {event.get("kind") for event in parsed["events"]}
        self.assertIn("agent_started", event_kinds)
        self.assertIn("agent_progress", event_kinds)
        self.assertIn("agent_completed", event_kinds)
        self.assertNotIn("tasks", parsed)

        parent_call = next(
            event for event in parsed["events"]
            if event.get("call_id") == "agent-call"
        )
        self.assertNotIn("agent", parent_call)

        child_events = [
            event for event in parsed["events"]
            if event.get("agent") == "Solve Wordle [agent-1]"
        ]
        self.assertTrue(child_events)
        self.assertTrue(any(event["type"] == "reasoning" for event in child_events))
        self.assertTrue(any(event.get("call_id") == "child-call" for event in child_events))
        child_result = next(
            event for event in child_events if event.get("call_id") == "child-call"
        )
        self.assertEqual(child_result["turn"], 2)

    def test_agent_loop_renderer_handles_generic_subagent_events(self):
        agent_loop = {
            "schema_version": 1,
            "backend": "future_harness",
            "events": [{
                "sequence": 1,
                "type": "message",
                "agent": "worker-1",
                "kind": "started",
                "content": {"description": "Solve a subtask"},
            }],
        }

        rendered = build_agent_loop_html(
            agent_loop,
            Path("/tmp/future-harness/agent_loop.json"),
        )

        self.assertIn("Agent Loop Transcript — future_harness", rendered)
        self.assertIn("agent-event", rendered)
        self.assertIn("agent=worker-1", rendered)
        self.assertNotIn("Claude", rendered)

    def test_hermes_trace_parser_standardizes_verbose_cli_output(self):
        trace = "\n".join([
            "agent_loop_instruction_start",
            "Play through MCP tools.",
            "agent_loop_instruction_end",
            "hermes_chat_command:",
            "hermes chat --provider openrouter --model test-model --yolo -q <instruction>",
            "hermes_chat_stdout:",
            "Query: Play through MCP tools.",
            "Initializing agent...",
            "🤖 AI Agent initialized with model: test-model",
            "🛠️  Final tool selection (2 tools): web_search, mcp__clem_game__start_game",
            "┌─ Reasoning ─────────┐",
            "I should start the game.",
            "└─────────────────────┘",
            "  📞 Tool 1: mcp__clem_game__start_game([])",
            "     Args: {}",
            "  ┊ ⚡ preparing mcp__clem_game__start_game…",
            "  ✅ Tool 1 completed in 0.10s",
            "     Result: {\"structuredContent\": {\"context\": {\"role\": \"user\", \"content\": \"Game prompt\"}, \"done\": false}}",
            "hermes_chat_stderr:",
            "success: True",
        ])

        with tempfile.TemporaryDirectory() as directory:
            episode_dir = Path(directory)
            (episode_dir / "agent_trace.log").write_text(trace, encoding="utf-8")
            parsed = HermesHarness.parse_agent_trace(episode_dir)

        event_types = [event["type"] for event in parsed["events"]]
        self.assertEqual(parsed["backend"], "hermes")
        self.assertIn("instruction", event_types)
        self.assertIn("reasoning", event_types)
        self.assertIn("tool_call", event_types)
        self.assertIn("tool_result", event_types)
        self.assertEqual(parsed["runtime"]["model"], "test-model")
        self.assertTrue(parsed["result"]["success"])
        self.assertEqual(parsed["capture"]["tool_definitions"]["count"], 2)

    def test_hermes_trace_parser_prefers_session_export(self):
        trace = "\n".join([
            "agent_loop_instruction_start",
            "Play through MCP tools.",
            "agent_loop_instruction_end",
            "hermes_chat_stdout:",
            "Query: Play through MCP tools.",
            "Initializing agent...",
            "hermes_chat_stderr:",
            "success: True",
        ])
        session = {
            "id": "session-1",
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Play through MCP tools."},
                {
                    "role": "assistant",
                    "content": "I will verify this.",
                    "reasoning": "I should search.",
                    "tool_calls": [{
                        "id": "call-1",
                        "function": {
                            "name": "web_search",
                            "arguments": "{\"query\": \"example\"}"
                        }
                    }]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "tool_name": "web_search",
                    "content": "search result"
                }
            ]
        }

        with tempfile.TemporaryDirectory() as directory:
            episode_dir = Path(directory)
            (episode_dir / "agent_trace.log").write_text(trace, encoding="utf-8")
            (episode_dir / "hermes_session_export.jsonl").write_text(
                json.dumps(session) + "\n",
                encoding="utf-8",
            )
            parsed = HermesHarness.parse_agent_trace(episode_dir)

        event_types = [event["type"] for event in parsed["events"]]
        self.assertIn("reasoning", event_types)
        self.assertIn("tool_preamble", event_types)
        self.assertIn("tool_call", event_types)
        self.assertIn("tool_result", event_types)
        self.assertEqual(parsed["runtime"]["session_id"], "session-1")
        self.assertEqual(
            parsed["capture"]["semantic_events"]["source"],
            "hermes_session_export",
        )

    def test_hermes_trace_parser_recovers_streamed_wire_events(self):
        instruction = "Play through MCP tools."
        request_one = {
            "messages": [
                {"role": "system", "content": "Native Hermes instruction."},
                {"role": "user", "content": instruction},
            ]
        }
        call_id = "call-1"
        response_one = [
            {
                "choices": [{
                    "delta": {
                        "role": "assistant",
                        "reasoning": "I should inspect the game.",
                    },
                    "finish_reason": None,
                }]
            },
            {
                "choices": [{
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [{
                            "index": 0,
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "mcp__game__start_game",
                                "arguments": "",
                            },
                        }],
                    },
                    "finish_reason": None,
                }]
            },
            {
                "choices": [{
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [{
                            "index": 0,
                            "function": {"arguments": "{}"},
                        }],
                    },
                    "finish_reason": "tool_calls",
                }]
            },
        ]
        request_two = {
            "messages": request_one["messages"] + [
                {
                    "role": "assistant",
                    "reasoning": "I should inspect the game.",
                    "tool_calls": [{
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "mcp__game__start_game",
                            "arguments": "{}",
                        },
                    }],
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": "mcp__game__start_game",
                    "content": [{
                        "type": "text",
                        "text": "Game prompt",
                    }, {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,AAAA"},
                    }],
                },
            ]
        }
        response_two = [{
            "choices": [{
                "delta": {
                    "role": "assistant",
                    "content": "I can answer now.",
                },
                "finish_reason": "stop",
            }]
        }]
        trace_lines = [
            "agent_loop_instruction_start",
            instruction,
            "agent_loop_instruction_end",
            "hermes_chat_stdout:",
            "┌─ Reasoning ─────────┐",
            "I should inspect the game.",
            "└─────────────────────┘",
            "hermes_chat_stderr:",
        ]

        for number, (request, response) in enumerate(
                ((request_one, response_one), (request_two, response_two)), start=1):
            trace_lines.extend([
                f"raw_upstream_request_{number}_start",
                "path: /api/v1/chat/completions",
                json.dumps(request),
                f"raw_upstream_request_{number}_end",
                f"raw_upstream_response_{number}_start",
                "path: /api/v1/chat/completions",
                "status: 200",
                "content_type: text/event-stream",
                "content_encoding: identity",
                *[f"data: {json.dumps(chunk)}\n" for chunk in response],
                "data: [DONE]\n",
                f"raw_upstream_response_{number}_end",
            ])

        with tempfile.TemporaryDirectory() as directory:
            episode_dir = Path(directory)
            (episode_dir / "agent_trace.log").write_text(
                "\n".join(trace_lines),
                encoding="utf-8",
            )
            parsed = HermesHarness.parse_agent_trace(episode_dir)

        semantic_events = [
            event for event in parsed["events"]
            if event["type"] in {
                "reasoning", "tool_call", "tool_result", "assistant_text"
            }
        ]
        self.assertEqual(
            [event["type"] for event in semantic_events],
            ["reasoning", "tool_call", "tool_result", "assistant_text"],
        )
        self.assertEqual(semantic_events[1]["name"], "mcp__game__start_game")
        self.assertEqual(semantic_events[1]["arguments"], {})
        self.assertIn("Game prompt", semantic_events[2]["content"])
        self.assertIn("image content supplied", semantic_events[2]["content"])
        self.assertEqual(
            parsed["capture"]["semantic_events"]["source"],
            "chat_completions_wire",
        )

        with tempfile.TemporaryDirectory() as directory:
            episode_dir = Path(directory)
            (episode_dir / "agent_trace.log").write_text(
                "\n".join(trace_lines).replace(
                    "path: /api/v1/chat/completions", "path: /v1/chat/completions",
                ),
                encoding="utf-8",
            )
            local_parsed = HermesHarness.parse_agent_trace(episode_dir)
        self.assertEqual(
            local_parsed["capture"]["semantic_events"],
            parsed["capture"]["semantic_events"],
        )
        self.assertEqual(
            [e for e in local_parsed["events"] if e["type"] in {
                "reasoning", "tool_call", "tool_result", "assistant_text",
            }],
            semantic_events,
        )

    def test_agent_loop_renderer_omits_embedded_media_and_binary_wire_bodies(self):
        agent_loop = {
            "schema_version": 1,
            "backend": "future_harness",
            "events": [{
                "sequence": 1,
                "type": "tool_result",
                "content": [{
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,SECRETBYTES"},
                }],
            }, {
                "sequence": 2,
                "type": "model_response",
                "turn": 1,
                "content_encoding": "gzip",
                "raw": "\x1f\x00SECRET-BINARY",
            }],
        }

        rendered = build_agent_loop_html(
            agent_loop,
            Path("/tmp/future-harness/agent_loop.json"),
        )

        self.assertNotIn("SECRETBYTES", rendered)
        self.assertNotIn("SECRET-BINARY", rendered)
        self.assertIn("embedded image data omitted", rendered)
        self.assertIn("compressed or binary wire body omitted", rendered)

    def test_hermes_trace_parser_pairs_concurrent_tools(self):
        trace = "\n".join([
            "hermes_chat_stdout:",
            "┌─ Reasoning ─────────┐",
            "I should search twice.",
            "└─────────────────────┘",
            "  ⚡ Concurrent: 2 tool calls — web_search, web_search",
            "  📞 Tool 1: web_search(['query'])",
            "     Args: {\"query\": \"first\"}",
            "  📞 Tool 2: web_search(['query'])",
            "     Args: {\"query\": \"second\"}",
            "  ┊ 🔍 search first",
            "  ✅ Tool 1 completed in 0.10s",
            "     Result: {\"output\": \"first result\"}",
            "  ┊ 🔍 search second",
            "  ✅ Tool 2 completed in 0.20s",
            "     Result: {\"output\": \"second result\"}",
            "hermes_chat_stderr:",
            "success: True",
        ])

        with tempfile.TemporaryDirectory() as directory:
            episode_dir = Path(directory)
            (episode_dir / "agent_trace.log").write_text(trace, encoding="utf-8")
            parsed = HermesHarness.parse_agent_trace(episode_dir)

        calls = [event for event in parsed["events"] if event["type"] == "tool_call"]
        results = [event for event in parsed["events"] if event["type"] == "tool_result"]
        self.assertEqual([event["arguments"]["query"] for event in calls], ["first", "second"])
        self.assertEqual([event["call_id"] for event in results], ["hermes-call-1", "hermes-call-2"])

    def test_openclaw_trace_parser_standardizes_native_session(self):
        stdout = {
            "payloads": [{"text": "DONE"}],
            "meta": {
                "durationMs": 50,
                "aborted": False,
                "agentMeta": {
                    "sessionId": "session-1",
                    "provider": "openrouter",
                    "model": "test-model",
                    "usage": {"input": 10, "output": 5}
                },
                "systemPromptReport": {
                    "systemPrompt": {"chars": 123, "hash": "prompt-hash"}
                }
            }
        }
        session = [
            {"type": "session", "id": "session-1", "cwd": "/workspace"},
            {
                "type": "model_change",
                "provider": "openrouter",
                "modelId": "test-model"
            },
            {"type": "thinking_level_change", "thinkingLevel": "high"},
            {
                "type": "message",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "Play through MCP tools."}]
                }
            },
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "I should search."},
                        {"type": "text", "text": "I will verify this."},
                        {
                            "type": "toolCall",
                            "id": "call-1",
                            "name": "web_search",
                            "arguments": {"query": "example"}
                        }
                    ]
                }
            },
            {
                "type": "message",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "call-1",
                    "toolName": "web_search",
                    "content": [{"type": "text", "text": "search result"}]
                }
            },
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "DONE"}]
                }
            }
        ]
        trajectory = [{
            "type": "context.compiled",
            "data": {
                "systemPrompt": {
                    "truncated": True,
                    "originalChars": 123
                },
                "prompt": "Play through MCP tools.",
                "tools": [{"name": "trajectory_tool"}]
            }
        }]
        request = {
            "messages": [
                {"role": "system", "content": "Native OpenClaw instruction."},
                {"role": "user", "content": "Play through MCP tools."}
            ],
            "tools": [{"type": "function", "function": {"name": "web_search"}}]
        }
        trace_lines = [
            "agent_loop_instruction_start",
            "Play through MCP tools.",
            "agent_loop_instruction_end",
            "openclaw_agent_stdout:",
            json.dumps(stdout),
            "openclaw_agent_stderr:",
            "openclaw_session: /tmp/session-1.jsonl",
            *[json.dumps(record) for record in session],
            "openclaw_session: /tmp/session-1.trajectory.jsonl",
            *[json.dumps(record) for record in trajectory],
            "raw_upstream_request_1_start",
            "path: /api/v1/chat/completions",
            json.dumps(request),
            "raw_upstream_request_1_end",
            "raw_upstream_response_1_start",
            "path: /api/v1/chat/completions",
            "status: 200",
            "content_type: application/json",
            "content_encoding: identity",
            json.dumps({"choices": []}),
            "raw_upstream_response_1_end"
        ]

        with tempfile.TemporaryDirectory() as directory:
            episode_dir = Path(directory)
            (episode_dir / "agent_trace.log").write_text(
                "\n".join(trace_lines),
                encoding="utf-8",
            )
            parsed = OpenClawHarness.parse_agent_trace(episode_dir)

        event_types = [event["type"] for event in parsed["events"]]
        instruction_kinds = [event.get("kind") for event in parsed["events"]
                             if event["type"] == "instruction"]
        self.assertEqual(parsed["backend"], "openclaw")
        self.assertEqual(instruction_kinds, ["native_harness", "agent_loop"])
        self.assertIn("reasoning", event_types)
        self.assertIn("tool_preamble", event_types)
        self.assertIn("tool_call", event_types)
        self.assertIn("tool_result", event_types)
        self.assertIn("assistant_text", event_types)
        self.assertEqual(parsed["runtime"]["thinking_level"], "high")
        self.assertEqual(parsed["result"]["final_text"], "DONE")
        self.assertEqual(parsed["capture"]["tool_definitions"]["source"], "wire_request")
        self.assertEqual(parsed["capture"]["model_requests"]["count"], 1)
        self.assertEqual(parsed["capture"]["model_responses"]["count"], 1)

    def test_openclaw_trace_parser_treats_post_game_abort_as_termination(self):
        session = [
            {
                "type": "message",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "Play through MCP tools."}]
                }
            },
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [{
                        "type": "toolCall",
                        "id": "call-1",
                        "name": "clem_game__submit_response",
                        "arguments": {"response": "GUESS: answer"}
                    }]
                }
            },
            {
                "type": "message",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "call-1",
                    "toolName": "clem_game__submit_response",
                    "content": [{"type": "text", "text": "done: true"}],
                    "details": {"structuredContent": {"done": True}}
                }
            },
            {
                "type": "custom",
                "customType": "openclaw:prompt-error",
                "data": {"error": "This operation was aborted | 20"}
            },
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "stopReason": "aborted",
                    "content": [{"type": "thinking", "thinking": "Partial final reasoning."}]
                }
            }
        ]
        trajectory = [{
            "type": "session.ended",
            "data": {
                "status": "error",
                "aborted": True,
                "externalAbort": True,
                "promptError": "This operation was aborted | 20"
            }
        }]
        trace_lines = [
            "openclaw_session: /tmp/session-1.jsonl",
            *[json.dumps(record) for record in session],
            "openclaw_session: /tmp/session-1.trajectory.jsonl",
            *[json.dumps(record) for record in trajectory]
        ]

        with tempfile.TemporaryDirectory() as directory:
            episode_dir = Path(directory)
            (episode_dir / "agent_trace.log").write_text(
                "\n".join(trace_lines),
                encoding="utf-8",
            )
            parsed = OpenClawHarness.parse_agent_trace(episode_dir)

        meaningful_events = [event for event in parsed["events"]
                             if event["type"] in {"reasoning", "termination", "error"}]
        self.assertEqual(
            [event["type"] for event in meaningful_events],
            ["reasoning", "termination"],
        )
        self.assertEqual(parsed["result"]["status"], "completed")
        self.assertTrue(parsed["result"]["success"])
        self.assertEqual(
            parsed["result"]["terminal_reason"],
            "terminated_after_game_completion",
        )

    def test_openclaw_trace_parser_recovers_partial_wire_history(self):
        instruction = "Play through MCP tools."
        system_message = {"role": "system", "content": "Native OpenClaw instruction."}
        user_message = {"role": "user", "content": instruction}
        search_call = {
            "role": "assistant",
            "content": "I will verify this.",
            "reasoning": "A search could help.",
            "tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "web_search",
                    "arguments": json.dumps({"query": "example"})
                }
            }]
        }
        search_result = {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": "search result"
        }
        submit_call = {
            "role": "assistant",
            "content": None,
            "reasoning": "I can now answer.",
            "tool_calls": [{
                "id": "call-2",
                "type": "function",
                "function": {
                    "name": "clem_game__submit_response",
                    "arguments": json.dumps({"response": "GUESS: answer"})
                }
            }]
        }
        submit_result = {
            "role": "tool",
            "tool_call_id": "call-2",
            "content": json.dumps({"done": True})
        }
        requests = [
            [system_message, user_message],
            [system_message, user_message, search_call, search_result],
            [system_message, user_message, search_call, search_result,
             submit_call, submit_result]
        ]
        trace_lines = [
            "agent_loop_instruction_start",
            instruction,
            "agent_loop_instruction_end"
        ]

        for request_number, messages in enumerate(requests, start=1):
            trace_lines.extend([
                f"raw_upstream_request_{request_number}_start",
                "path: /api/v1/chat/completions",
                json.dumps({"messages": messages}),
                f"raw_upstream_request_{request_number}_end"
            ])

        with tempfile.TemporaryDirectory() as directory:
            episode_dir = Path(directory)
            (episode_dir / "agent_trace.log").write_text(
                "\n".join(trace_lines),
                encoding="utf-8",
            )
            parsed = OpenClawHarness.parse_agent_trace(episode_dir)

        calls = [event for event in parsed["events"] if event["type"] == "tool_call"]
        results = [event for event in parsed["events"] if event["type"] == "tool_result"]
        reasoning = [event for event in parsed["events"] if event["type"] == "reasoning"]
        self.assertEqual([event["name"] for event in calls], [
            "web_search",
            "clem_game__submit_response"
        ])
        self.assertEqual([event["call_id"] for event in results], ["call-1", "call-2"])
        self.assertEqual(len(reasoning), 2)
        self.assertEqual(parsed["capture"]["semantic_events"]["status"], "partial")
        self.assertEqual(parsed["capture"]["semantic_events"]["source"], "wire_request")

    def test_bridge_exposes_start_and_submit_without_get_state(self):
        client = unittest.mock.MagicMock()

        with patch(
            "clemagents.mcp.bridge.OpenEnvMCPClient",
            return_value=client,
        ):
            bridge = create_mcp_bridge("http://example.invalid/mcp")

        tools = asyncio.run(bridge.list_tools())

        self.assertEqual(
            [tool.name for tool in tools], ["start_game", "submit_response"]
        )

    def test_bridge_aborts_repeated_start_game(self):
        client = unittest.mock.MagicMock()
        client.call_tool.side_effect = [
            {
                "context": {"role": "user", "content": "Initial game message"},
                "reward": None,
                "done": False,
                "metadata": {}
            },
            {
                "context": {"role": "user", "content": "Initial game message"},
                "reward": -1.0,
                "done": False,
                "metadata": {}
            },
        ]

        with tempfile.TemporaryDirectory() as directory:
            completion_path = Path(directory) / "completed.json"

            with patch(
                "clemagents.mcp.bridge.OpenEnvMCPClient",
                return_value=client,
            ), patch.dict(
                "os.environ",
                {"GAME_COMPLETION_PATH": str(completion_path)},
            ):
                bridge = create_mcp_bridge("http://example.invalid/mcp")

                async def repeat_start_game():
                    await bridge.call_tool("start_game", {})
                    return await bridge.call_tool("start_game", {})

                result = asyncio.run(repeat_start_game())

            completion = json.loads(completion_path.read_text(encoding="utf-8"))

        self.assertEqual(
            result.structured_content,
            {"message": REPEATED_START_MESSAGE},
        )
        self.assertEqual(
            json.loads(result.content[0].text),
            {"message": REPEATED_START_MESSAGE},
        )
        self.assertEqual(
            client.call_tool.call_args_list,
            [
                unittest.mock.call("start_game", {}),
                unittest.mock.call(
                    "submit_response", {"response": CONTROL_FAILURE_RESPONSE}
                ),
            ],
        )
        client.close_session.assert_called_once_with()
        self.assertTrue(completion["done"])
        self.assertTrue(completion["control_failure"])

    def test_bridge_aborts_submit_response_before_start_game(self):
        client = unittest.mock.MagicMock()

        with tempfile.TemporaryDirectory() as directory:
            completion_path = Path(directory) / "completed.json"

            with patch(
                "clemagents.mcp.bridge.OpenEnvMCPClient",
                return_value=client,
            ), patch.dict(
                "os.environ",
                {"GAME_COMPLETION_PATH": str(completion_path)},
            ):
                bridge = create_mcp_bridge("http://example.invalid/mcp")
                result = asyncio.run(
                    bridge.call_tool("submit_response", {"response": "guess"})
                )

            completion = json.loads(completion_path.read_text(encoding="utf-8"))

        self.assertEqual(
            result.structured_content,
            {
                "message": (
                    "The episode was aborted because submit_response was called "
                    "before start_game."
                )
            },
        )
        self.assertEqual(
            json.loads(result.content[0].text),
            {
                "message": (
                    "The episode was aborted because submit_response was called "
                    "before start_game."
                )
            },
        )
        client.call_tool.assert_not_called()
        self.assertTrue(completion["control_failure"])

    def test_bridge_writes_completion_marker(self):
        client = unittest.mock.MagicMock()
        client.call_tool.side_effect = [
            {
                "context": {"role": "user", "content": "Initial game message"},
                "reward": None,
                "done": False,
                "metadata": {}
            },
            {
                "context": {"role": "user", "content": "Final game message"},
                "reward": 1.0,
                "done": True,
                "metadata": {}
            },
        ]

        with tempfile.TemporaryDirectory() as directory:
            completion_path = Path(directory) / "completed.json"

            with patch(
                "clemagents.mcp.bridge.OpenEnvMCPClient",
                return_value=client,
            ), patch.dict(
                "os.environ",
                {"GAME_COMPLETION_PATH": str(completion_path)},
            ):
                bridge = create_mcp_bridge("http://example.invalid/mcp")

                async def finish_game():
                    start_result = await bridge.call_tool("start_game", {})
                    final_result = await bridge.call_tool(
                        "submit_response", {"response": "guess"}
                    )
                    return start_result, final_result

                start_result, final_result = asyncio.run(finish_game())

            completion = json.loads(completion_path.read_text(encoding="utf-8"))

        self.assertTrue(completion["done"])
        self.assertFalse(completion["control_failure"])
        self.assertEqual(completion["reward"], 1.0)
        self.assertEqual(
            start_result.structured_content,
            {"message": "Initial game message"},
        )
        self.assertEqual(
            json.loads(start_result.content[0].text),
            {"message": "Initial game message"},
        )
        self.assertEqual(
            final_result.structured_content,
            {"message": "Final game message"},
        )
        self.assertEqual(
            json.loads(final_result.content[0].text),
            {"message": "Final game message"},
        )

    def test_bridge_preserves_text_format_and_adds_mcp_image_content(self):
        client = unittest.mock.MagicMock()
        client.call_tool.return_value = {
            "context": {
                "role": "user",
                "content": "Identify this location.",
                "image": [{
                    "data": "aW1hZ2UtYnl0ZXM=",
                    "mimeType": "image/jpeg",
                }],
            },
            "reward": None,
            "done": False,
            "metadata": {},
        }

        with patch(
            "clemagents.mcp.bridge.OpenEnvMCPClient",
            return_value=client,
        ):
            bridge = create_mcp_bridge("http://example.invalid/mcp")
            result = asyncio.run(bridge.call_tool("start_game", {}))

        self.assertEqual(
            json.loads(result.content[0].text),
            {"message": "Identify this location."},
        )
        self.assertEqual(result.content[1].type, "image")
        self.assertEqual(result.content[1].data, "aW1hZ2UtYnl0ZXM=")
        self.assertEqual(result.content[1].mimeType, "image/jpeg")
        self.assertEqual(
            result.structured_content,
            {"message": "Identify this location."},
        )

    def test_bridge_materializes_only_sequential_observation_images(self):
        client = unittest.mock.MagicMock()
        client.call_tool.side_effect = [
            {
                "context": {
                    "role": "user",
                    "content": "First view.",
                    "image": [{
                        "data": "Zmlyc3QtaW1hZ2U=",
                        "mimeType": "image/jpeg",
                    }],
                },
                "reward": None,
                "done": False,
                "metadata": {},
            },
            {
                "context": {
                    "role": "user",
                    "content": "Second view.",
                    "image": [{
                        "data": "c2Vjb25kLWltYWdl",
                        "mimeType": "image/png",
                    }],
                },
                "reward": None,
                "done": False,
                "metadata": {},
            },
        ]

        with tempfile.TemporaryDirectory() as directory:
            observation_dir = Path(directory) / "game_observations"

            with patch(
                "clemagents.mcp.bridge.OpenEnvMCPClient",
                return_value=client,
            ), patch.dict(
                "os.environ",
                {"GAME_OBSERVATION_DIR": str(observation_dir)},
            ):
                bridge = create_mcp_bridge("http://example.invalid/mcp")

                async def advance_game():
                    first = await bridge.call_tool("start_game", {})
                    second = await bridge.call_tool(
                        "submit_response", {"response": "next"}
                    )
                    return first, second

                first_result, second_result = asyncio.run(advance_game())

            first_path = observation_dir / "observation_000" / "image_00.jpg"
            second_path = observation_dir / "observation_001" / "image_00.png"
            future_path = observation_dir / "observation_002"

            self.assertEqual(first_path.read_bytes(), b"first-image")
            self.assertEqual(second_path.read_bytes(), b"second-image")
            self.assertFalse(future_path.exists())

            self.assertEqual(first_result.content[1].type, "image")
            self.assertEqual(first_result.content[2].type, "text")
            self.assertIn(str(first_path), first_result.content[2].text)
            self.assertNotIn(
                "resource_link",
                [block.type for block in first_result.content],
            )

            self.assertEqual(second_result.content[1].type, "image")
            self.assertEqual(second_result.content[2].type, "text")
            self.assertIn(str(second_path), second_result.content[2].text)
            self.assertNotIn(
                "resource_link",
                [block.type for block in second_result.content],
            )

    def test_hermes_timeout_is_a_failed_episode_not_an_exception(self):
        successful_setup = subprocess.CompletedProcess([], 0, "", "")

        with patch(
            "clemagents.adapters.hermes.load_model_connection",
            return_value=None,
        ), patch(
            "clemagents.adapters.hermes.subprocess.run",
            return_value=successful_setup,
        ), patch(
            "clemagents.adapters.hermes.run_process_until_game_complete",
            side_effect=subprocess.TimeoutExpired(
                ["hermes", "chat"],
                1,
                output="partial Hermes output",
                stderr="",
            ),
        ):
            result = HermesHarness(model="test-model").run_episode(
                "Play the game."
            )

        self.assertFalse(result.success)
        self.assertEqual(
            result.metadata["runtime_error"], "Hermes timed out after 1s"
        )

    def test_hermes_cli_exit_before_game_done_is_not_success(self):
        successful_setup = subprocess.CompletedProcess([], 0, "", "")
        incomplete_chat = subprocess.CompletedProcess(
            [],
            0,
            "Tool call: mcp__clem_game__start_game",
            "",
        )

        with patch(
            "clemagents.adapters.hermes.load_model_connection",
            return_value=None,
        ), patch(
            "clemagents.adapters.hermes.subprocess.run",
            return_value=successful_setup,
        ), patch(
            "clemagents.adapters.hermes.run_process_until_game_complete",
            return_value=(incomplete_chat, False),
        ):
            result = HermesHarness(model="test-model").run_episode(
                "Play the game."
            )

        self.assertFalse(result.success)
        self.assertFalse(result.metadata["game_completed"])
        self.assertEqual(
            result.metadata["runtime_error"],
            "Hermes ended before clem_game reported done=true",
        )

    def test_hermes_records_both_provider_routes_without_verbose_printer(self):
        received = []
        response_body = {
            "choices": [{"message": {
                "role": "assistant", "content": "fixture response",
                "reasoning_content": "fixture reasoning",
            }, "finish_reason": "stop"}],
        }

        class UpstreamHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                received.append(json.loads(self.rfile.read(length)))
                body = json.dumps(response_body).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args):
                return

        upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
        upstream_thread = Thread(target=upstream.serve_forever, daemon=True)
        upstream_thread.start()
        target = f"http://127.0.0.1:{upstream.server_port}/v1"
        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Inspect this image."},
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64,ZmFrZQ==",
                }},
            ]}],
            "tools": [{"type": "function", "function": {
                "name": "mcp__game__submit_response",
                "parameters": {"type": "object", "properties": {}},
            }}],
        }

        try:
            for provider, backend, endpoint_variable in (
                ("openai-api", "openai_compatible", "OPENAI_BASE_URL"),
                ("openrouter", "openrouter", "OPENROUTER_BASE_URL"),
            ):
                with self.subTest(provider=provider), tempfile.TemporaryDirectory() as directory:
                    config = {}
                    connection = {
                        "backend": backend, "provider": provider,
                        "model": "test-model", "base_url": target,
                        "env": {endpoint_variable: target},
                    }

                    def setup(command, **kwargs):
                        if command[:3] == ["hermes", "config", "set"]:
                            config[command[3]] = command[4]
                        return subprocess.CompletedProcess(command, 0, "", "")

                    def chat(command, **kwargs):
                        self.assertNotIn("--verbose", command)
                        self.assertEqual(config["model.provider"], provider)
                        self.assertEqual(config["model.default"], "test-model")
                        self.assertEqual(config["display.show_reasoning"], "true")
                        self.assertEqual(config["display.tool_progress"], "verbose")
                        endpoint = os.environ[endpoint_variable]
                        self.assertNotEqual(endpoint, target)
                        self.assertEqual(config["model.base_url"], endpoint)
                        response = requests.post(
                            f"{endpoint}/chat/completions", json=payload, timeout=5,
                        )
                        response.raise_for_status()
                        self.assertEqual(response.json(), response_body)
                        return subprocess.CompletedProcess(
                            command, 0, "Tool call: mcp__game__start_game", "",
                        ), False

                    with patch(
                        "clemagents.adapters.hermes.load_model_connection",
                        return_value=connection,
                    ), patch(
                        "clemagents.adapters.hermes.subprocess.run",
                        side_effect=setup,
                    ), patch(
                        "clemagents.adapters.hermes.run_process_until_game_complete",
                        side_effect=chat,
                    ):
                        result = HermesHarness(model="test-model").run_episode(
                            "Play the game.", output_dir=directory,
                        )

                    self.assertEqual(received[-1], payload)
                    trace_path = Path(result.artifacts["adapter_messages"])
                    trace = trace_path.read_text(encoding="utf-8")
                    self.assertIn("raw_upstream_request_", trace)
                    self.assertIn("raw_upstream_response_", trace)
                    (Path(directory) / "agent_trace.log").write_text(
                        trace, encoding="utf-8",
                    )
                    parsed = HermesHarness.parse_agent_trace(Path(directory))
                    self.assertEqual(parsed["capture"]["model_requests"]["count"], 1)
                    self.assertEqual(parsed["capture"]["model_responses"]["count"], 1)
                    self.assertEqual(parsed["capture"]["tool_definitions"]["status"], "complete")
        finally:
            upstream.shutdown()
            upstream.server_close()
            upstream_thread.join(timeout=5)

    def test_closed_openenv_session_removes_recovery_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            session_path = Path(directory) / "openenv_session.json"
            session_path.write_text(
                json.dumps({"session_id": "session-1"}),
                encoding="utf-8",
            )
            client = OpenEnvMCPClient("http://example.invalid/mcp")
            client.session_id = "session-1"

            with patch.object(client, "_request", return_value={}) as request, patch.dict(
                "os.environ",
                {"GAME_SESSION_PATH": str(session_path)},
            ):
                client.close_session()

            request.assert_called_once_with(
                "openenv/session/close",
                {"session_id": "session-1"},
            )
            self.assertIsNone(client.session_id)
            self.assertFalse(session_path.exists())

    def test_openclaw_rejects_mismatched_openrouter_connection(self):
        with self.assertRaisesRegex(ValueError, "canonical"):
            _validate_openclaw_model_connection({
                "backend": "openrouter",
                "model": "openai/gpt-5-mini",
                "env": {"OPENROUTER_API_KEY": "openrouter-test-key"},
            })

        with self.assertRaisesRegex(ValueError, "OPENROUTER_API_KEY"):
            _validate_openclaw_model_connection({
                "backend": "openrouter",
                "model": "openrouter/openai/gpt-5-mini",
                "env": {},
            })

    def test_openclaw_openrouter_resolution_uses_canonical_model(self):
        model_spec = {
            "model_name": "gpt-5-mini-openrouter",
            "backend": "openrouter",
            "model_id": "openai/gpt-5-mini",
        }
        with patch.object(
            model_connection, "_find_model_spec", return_value=model_spec
        ), patch.object(
            model_connection,
            "_openrouter_key_config",
            return_value={"api_key": "openrouter-test-key"},
        ):
            connection = OpenClawHarness.resolve_model_connection(
                "gpt-5-mini-openrouter"
            )

        self.assertEqual(connection["harness"], "openclaw")
        self.assertEqual(connection["backend"], "openrouter")
        self.assertEqual(
            connection["model"], "openrouter/openai/gpt-5-mini"
        )
        self.assertEqual(connection["upstream_model"], "openai/gpt-5-mini")
        self.assertEqual(connection["base_url"], "https://openrouter.ai/api/v1")
        self.assertEqual(connection["openclaw_provider"], "openrouter")
        self.assertEqual(
            connection["env"],
            {"OPENROUTER_API_KEY": "openrouter-test-key"},
        )

    def test_openclaw_openai_compatible_resolution_uses_registry_modalities(self):
        model_spec = {
            "model_name": "multimodal-test-model",
            "backend": "openai_compatible",
            "model_id": "local/multimodal-test-model",
            "context_size": "128k",
            "model_config": {
                "input_modalities": ["text", "image", "audio"],
                "extra_body": {"reasoning": {"effort": "high"}},
            },
        }
        key_config = {
            "api_key": "local-test-key",
            "base_url": "http://localhost:11434/v1",
        }

        with patch.object(
            model_connection, "_find_model_spec", return_value=model_spec
        ), patch.object(
            model_connection, "_openai_compatible_key_config", return_value=key_config
        ):
            connection = OpenClawHarness.resolve_model_connection(
                model_spec["model_name"]
            )

        provider = connection["openclaw_config_patch"]["models"]["providers"][
            "openai_compatible"
        ]
        self.assertEqual(provider["models"][0]["input"], ["text", "image"])

    def test_openclaw_openai_compatible_resolution_defaults_to_text_input(self):
        model_spec = {
            "model_name": "text-test-model",
            "backend": "openai_compatible",
            "model_id": "local/text-test-model",
            "model_config": {},
        }
        key_config = {
            "api_key": "local-test-key",
            "base_url": "http://localhost:11434/v1",
        }

        with patch.object(
            model_connection, "_find_model_spec", return_value=model_spec
        ), patch.object(
            model_connection, "_openai_compatible_key_config", return_value=key_config
        ):
            connection = OpenClawHarness.resolve_model_connection(
                model_spec["model_name"]
            )

        provider = connection["openclaw_config_patch"]["models"]["providers"][
            "openai_compatible"
        ]
        self.assertEqual(provider["models"][0]["input"], ["text"])

    def test_all_harnesses_resolve_openai_compatible_models(self):
        model_spec = {
            "model_name": "Qwen-test-without-reasoning",
            "backend": "openai_compatible",
            "model_id": "Qwen/Qwen-test",
            "context_size": "128k",
            "model_config": {
                "extra_body": {
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            },
        }
        key_config = {
            "api_key": "jarvis-test-key",
            "base_url": "https://jarvis.ling.uni-potsdam.de/api/v1",
            "verify_tls": False,
        }

        with patch.object(
            model_connection, "_find_model_spec", return_value=model_spec
        ), patch.object(
            model_connection, "_openai_compatible_key_config", return_value=key_config
        ):
            connections = {
                "claude_code": ClaudeCodeHarness.resolve_model_connection(
                    model_spec["model_name"]
                ),
                "codex": CodexHarness.resolve_model_connection(
                    model_spec["model_name"]
                ),
                "hermes": HermesHarness.resolve_model_connection(
                    model_spec["model_name"]
                ),
                "openclaw": OpenClawHarness.resolve_model_connection(
                    model_spec["model_name"]
                ),
            }

        for harness, connection in connections.items():
            self.assertEqual(connection["harness"], harness)
            self.assertEqual(connection["backend"], "openai_compatible")
            self.assertNotIn("tool_choice", connection)
            self.assertFalse(connection["verify_tls"])
            self.assertEqual(
                connection["request_body_overrides"],
                {"chat_template_kwargs": {"enable_thinking": False}},
            )

        self.assertEqual(connections["hermes"]["provider"], "openai-api")
        self.assertEqual(connections["claude_code"]["runtime_model"], "claude-sonnet-4-5")
        self.assertEqual(connections["claude_code"]["model"], "Qwen/Qwen-test")
        self.assertEqual(
            connections["openclaw"]["openclaw_provider"],
                "openai_compatible",
        )

    def test_agent_reasoning_off_disables_provider_reasoning(self):
        for effort in ("none", "off"):
            with self.subTest(effort=effort), tempfile.TemporaryDirectory() as directory:
                registry_path = Path(directory) / "agent_registry.json"
                registry_path.write_text(
                    json.dumps([
                        {
                            "agent_name": "test-agent",
                            "backend": "codex",
                            "agent_config": {
                                "clem_model": "test-model",
                                "reasoning_effort": effort,
                            },
                        }
                    ]),
                    encoding="utf-8",
                )

                with patch.object(
                    CodexHarness,
                    "resolve_model_connection",
                    return_value={
                        "request_body_overrides": {
                            "reasoning": {"enabled": True, "effort": "high"},
                        },
                    },
                ):
                    connection = model_connection.resolve_agent_model_connection(
                        "test-agent",
                        registry_path,
                    )

                self.assertEqual(
                    connection["request_body_overrides"]["reasoning"],
                    {"enabled": False},
                )

    def test_proxy_reasoning_off_removes_harness_effort(self):
        body, _ = _prepare_request_body(
            json.dumps({
                "model": "test-model",
                "reasoning": {
                    "enabled": True,
                    "effort": "high",
                    "summary": "auto",
                },
            }).encode("utf-8"),
            "/v1/responses",
            Path("/tmp/nonexistent-completion-marker"),
            {"reasoning": {"enabled": False}},
        )

        self.assertEqual(
            json.loads(body)["reasoning"],
            {"enabled": False},
        )

    def test_proxy_preserves_harness_tool_choice(self):
        received = []

        class UpstreamHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                received.append((self.path, json.loads(self.rfile.read(length))))
                response = b'{"ok":true}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)

            def log_message(self, format, *args):
                return

        upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
        upstream_thread = Thread(target=upstream.serve_forever, daemon=True)
        upstream_thread.start()

        try:
            with tempfile.TemporaryDirectory() as directory:
                completion_path = Path(directory) / "done.json"
                target = f"http://127.0.0.1:{upstream.server_port}/api/v1"

                with OpenAICompatibleProxy(
                    target,
                    completion_path,
                    {"chat_template_kwargs": {"enable_thinking": False}},
                ) as proxy:
                    openai_body = {
                        "model": "qwen",
                        "tools": [{"type": "function", "function": {"name": "start_game"}}],
                        "tool_choice": "auto",
                    }
                    requests.post(
                        f"{proxy.base_url}/chat/completions",
                        json=openai_body,
                        timeout=5,
                    ).raise_for_status()
                    requests.post(
                        f"{proxy.base_url}/messages",
                        json={
                            "model": "qwen",
                            "tools": openai_body["tools"],
                        },
                        timeout=5,
                    ).raise_for_status()

                    completion_path.write_text('{"done":true}', encoding="utf-8")
                    post_game_response = requests.post(
                        f"{proxy.base_url}/chat/completions",
                        json=openai_body,
                        timeout=5,
                    )
        finally:
            upstream.shutdown()
            upstream.server_close()
            upstream_thread.join(timeout=5)

        self.assertEqual(received[0][0], "/api/v1/chat/completions")
        self.assertEqual(received[0][1]["tool_choice"], "auto")
        self.assertEqual(
            received[0][1]["chat_template_kwargs"],
            {"enable_thinking": False},
        )
        self.assertNotIn("tool_choice", received[1][1])
        self.assertEqual(post_game_response.status_code, 409)
        self.assertEqual(len(received), 2)

    def test_proxy_rewrites_model_for_anthropic_messages(self):
        received = []

        class UpstreamHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                received.append((self.path, json.loads(self.rfile.read(length))))
                response = b'{"ok":true}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)

            def log_message(self, format, *args):
                return

        upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
        upstream_thread = Thread(target=upstream.serve_forever, daemon=True)
        upstream_thread.start()

        try:
            with tempfile.TemporaryDirectory() as directory:
                completion_path = Path(directory) / "done.json"
                target = f"http://127.0.0.1:{upstream.server_port}/api/v1"

                with OpenAICompatibleProxy(
                    target,
                    completion_path,
                    {"chat_template_kwargs": {"enable_thinking": False}},
                    upstream_model="Qwen/Qwen3.6-35B-A3B-FP8",
                ) as proxy:
                    payload = {
                        "model": "claude-sonnet-4-5",
                        "tools": [{"type": "function", "function": {"name": "start_game"}}],
                    }
                    requests.post(
                        f"{proxy.base_url}/messages",
                        json=payload,
                        timeout=5,
                    ).raise_for_status()
                    requests.post(
                        f"{proxy.base_url}/chat/completions",
                        json=payload,
                        timeout=5,
                    ).raise_for_status()
        finally:
            upstream.shutdown()
            upstream.server_close()
            upstream_thread.join(timeout=5)

        self.assertEqual(
            received[0][1]["model"],
            "Qwen/Qwen3.6-35B-A3B-FP8",
        )
        self.assertNotIn("chat_template_kwargs", received[0][1])
        self.assertEqual(
            received[1][1]["model"],
            "Qwen/Qwen3.6-35B-A3B-FP8",
        )
        self.assertEqual(
            received[1][1]["chat_template_kwargs"],
            {"enable_thinking": False},
        )






    def test_proxy_applies_generation_overrides_to_every_wire_protocol(self):
        openai_payload = {"model": "test", "messages": []}
        anthropic_payload = {"model": "test", "messages": []}

        with tempfile.TemporaryDirectory() as directory:
            completion_path = Path(directory) / "done.json"
            prepared_openai, _ = _prepare_request_body(
                json.dumps(openai_payload).encode(),
                "/v1/chat/completions",
                completion_path,
                {"reasoning": {"enabled": True}},
                generation_overrides={"temperature": 1.0},
            )
            prepared_anthropic, _ = _prepare_request_body(
                json.dumps(anthropic_payload).encode(),
                "/v1/messages",
                completion_path,
                {"reasoning": {"enabled": True}},
                generation_overrides={"temperature": 1.0},
            )

        prepared_openai_payload = json.loads(prepared_openai)
        prepared_anthropic_payload = json.loads(prepared_anthropic)
        self.assertEqual(prepared_openai_payload["temperature"], 1.0)
        self.assertEqual(prepared_anthropic_payload["temperature"], 1.0)
        self.assertEqual(
            prepared_openai_payload["reasoning"],
            {"enabled": True},
        )
        self.assertNotIn("reasoning", prepared_anthropic_payload)

    def test_generic_proxy_does_not_translate_tool_names(self):
        payload = {
            "model": "local-model",
            "input": [],
            "tools": [
                {
                    "type": "namespace",
                    "name": "mcp__game",
                    "tools": [{
                        "type": "function",
                        "name": "reset",
                        "parameters": {"type": "object"},
                    }],
                },
                {
                    "type": "namespace",
                    "name": "mcp__workspace",
                    "tools": [{
                        "type": "function",
                        "name": "reset",
                        "parameters": {"type": "object"},
                    }],
                },
            ],
        }

        with tempfile.TemporaryDirectory() as directory:
            body, exchange = _prepare_request_body(
                json.dumps(payload).encode(),
                "/v1/responses",
                Path(directory) / "done.json",
                {},
            )

        prepared = json.loads(body)
        self.assertEqual(
            prepared["tools"],
            payload["tools"],
        )
        self.assertIsNone(exchange)

    def test_proxy_blocks_new_inference_after_game_completion(self):
        upstream_requests = []

        class UpstreamHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                upstream_requests.append(self.path)
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, format, *args):
                return

        upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
        upstream_thread = Thread(target=upstream.serve_forever, daemon=True)
        upstream_thread.start()

        try:
            with tempfile.TemporaryDirectory() as directory:
                completion_path = Path(directory) / "done.json"
                completion_path.write_text('{"done":true}', encoding="utf-8")
                target = f"http://127.0.0.1:{upstream.server_port}/v1"

                with OpenAICompatibleProxy(target, completion_path) as proxy:
                    response = requests.post(
                        f"{proxy.base_url}/chat/completions",
                        json={"model": "test", "messages": []},
                        timeout=5,
                    )

            self.assertEqual(response.status_code, 409)
            self.assertEqual(response.json()["error"]["type"], "game_completed")
            self.assertEqual(upstream_requests, [])
        finally:
            upstream.shutdown()
            upstream.server_close()
            upstream_thread.join(timeout=5)


    def test_process_is_terminated_after_game_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            completion_path = Path(directory) / "done.json"
            script = (
                "from pathlib import Path; import time; "
                f"Path({str(completion_path)!r}).write_text('{{\"done\":true}}'); "
                "time.sleep(30)"
            )
            started_at = time.monotonic()
            result, terminated = run_process_until_game_complete(
                [sys.executable, "-c", script],
                completion_path=completion_path,
                completion_grace=0.01,
                timeout=5,
            )

        self.assertTrue(terminated)
        self.assertNotEqual(result.returncode, 0)
        self.assertLess(time.monotonic() - started_at, 2)

    def test_host_waits_for_artifacts_ready_after_game_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            shared_dir = Path(directory) / "shared"
            shared_dir.mkdir()
            keys_path = Path(directory) / "key.json"
            registry_path = Path(directory) / "agent_registry.json"
            keys_path.write_text("{}", encoding="utf-8")
            registry_path.write_text("[]", encoding="utf-8")
            completion_path = shared_dir / "game_completion.json"
            ready_path = shared_dir / "artifacts_ready.json"
            artifact_dir = shared_dir / "artifacts" / "test-agent" / "run-1"
            script = "\n".join([
                "import json, time",
                "from pathlib import Path",
                f"completion = Path({str(completion_path)!r})",
                f"ready = Path({str(ready_path)!r})",
                f"artifacts = Path({str(artifact_dir)!r})",
                "completion.write_text(json.dumps({'done': True, 'source': 'test'}))",
                "print('game done', flush=True)",
                "time.sleep(0.25)",
                "artifacts.mkdir(parents=True)",
                "(artifacts / 'native_session.jsonl').write_text('native artifact')",
                "ready.write_text(json.dumps({",
                "    'ready': True,",
                "    'status': 'complete',",
                "    'artifact_directory': 'artifacts/test-agent/run-1',",
                "}))",
                "print('artifacts ready', flush=True)",
                "time.sleep(0.05)",
            ])
            actual_popen = subprocess.Popen

            def simulated_container(_command, **kwargs):
                return actual_popen([sys.executable, "-c", script], **kwargs)

            with patch(
                "clemagents.run_pipeline.utils.KEYS_PATH",
                keys_path,
            ), patch(
                "clemagents.run_pipeline.utils.REGISTRY_PATH",
                registry_path,
            ), patch(
                "clemagents.run_pipeline.utils.subprocess.Popen",
                side_effect=simulated_container,
            ):
                trace, returncode, metadata = run_docker_episode(
                    experiment_name="test",
                    game_id=1,
                    agent_name="test-agent",
                    shared_state_dir=shared_dir,
                    episode_timeout=5,
                    artifact_finalization_timeout=2,
                    artifact_exit_grace=1,
                )

            self.assertEqual(returncode, 0)
            self.assertIn("game done", trace)
            self.assertIn("artifacts ready", trace)
            self.assertTrue(metadata["artifacts_ready"])
            self.assertFalse(metadata["artifact_finalization_timed_out"])
            self.assertEqual(
                metadata["agent_artifact_relative_path"],
                "artifacts/test-agent/run-1",
            )
            self.assertLess(
                metadata["episode_elapsed_seconds"],
                metadata["container_elapsed_seconds"],
            )
            self.assertTrue((artifact_dir / "native_session.jsonl").exists())

    def test_host_bounds_artifact_finalization_after_game_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            shared_dir = Path(directory) / "shared"
            shared_dir.mkdir()
            keys_path = Path(directory) / "key.json"
            registry_path = Path(directory) / "agent_registry.json"
            keys_path.write_text("{}", encoding="utf-8")
            registry_path.write_text("[]", encoding="utf-8")
            completion_path = shared_dir / "game_completion.json"
            script = "\n".join([
                "import json, time",
                "from pathlib import Path",
                f"Path({str(completion_path)!r}).write_text(json.dumps({{'done': True}}))",
                "print('game done', flush=True)",
                "time.sleep(30)",
            ])
            actual_popen = subprocess.Popen

            def simulated_container(_command, **kwargs):
                return actual_popen([sys.executable, "-c", script], **kwargs)

            def stop_simulated_container(process, _cidfile_path):
                process.terminate()

            with patch(
                "clemagents.run_pipeline.utils.KEYS_PATH",
                keys_path,
            ), patch(
                "clemagents.run_pipeline.utils.REGISTRY_PATH",
                registry_path,
            ), patch(
                "clemagents.run_pipeline.utils.subprocess.Popen",
                side_effect=simulated_container,
            ), patch(
                "clemagents.run_pipeline.utils._stop_docker_container",
                side_effect=stop_simulated_container,
            ):
                trace, returncode, metadata = run_docker_episode(
                    experiment_name="test",
                    game_id=1,
                    agent_name="test-agent",
                    shared_state_dir=shared_dir,
                    episode_timeout=5,
                    artifact_finalization_timeout=0.15,
                    artifact_exit_grace=0,
                )

            self.assertNotEqual(returncode, 0)
            self.assertIn("host_artifact_finalization_timeout", trace)
            self.assertTrue(metadata["artifact_finalization_timed_out"])
            self.assertFalse(metadata["artifacts_ready"])

    def test_openclaw_passes_openrouter_key_only_through_process_environment(self):
        connection = {
            "backend": "openrouter",
            "model": "openrouter/openai/gpt-5-mini",
            "upstream_model": "openai/gpt-5-mini",
            "base_url": "https://openrouter.ai/api/v1",
            "openclaw_provider": "openrouter",
            "env": {"OPENROUTER_API_KEY": "openrouter-test-key"},
        }
        calls = []
        observed_api_keys = []

        def fake_run(command, **kwargs):
            calls.append((command, kwargs.get("input")))
            observed_api_keys.append(os.environ.get("OPENROUTER_API_KEY"))
            return subprocess.CompletedProcess(command, 0, "", "")

        def fake_agent_run(command, **kwargs):
            calls.append((command, None))
            observed_api_keys.append(os.environ.get("OPENROUTER_API_KEY"))
            kwargs["completion_path"].write_text(
                '{"done":true,"control_failure":false}',
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(command, 0, '"submit_response"', ""), False

        with tempfile.TemporaryDirectory() as directory, patch(
            "clemagents.adapters.openclaw.load_model_connection",
            return_value=connection,
        ), patch(
            "clemagents.adapters.openclaw.subprocess.run",
            side_effect=fake_run,
        ), patch(
            "clemagents.adapters.openclaw.run_process_until_game_complete",
            side_effect=fake_agent_run,
        ), patch(
            "clemagents.adapters.openclaw.DUCKDUCKGO_PLUGIN_PATH",
            Path(directory),
        ), patch(
            "clemagents.adapters.openclaw._openclaw_gateway",
            return_value=nullcontext(),
        ) as gateway:
            (Path(directory) / "openclaw.plugin.json").write_text("{}", encoding="utf-8")
            result = OpenClawHarness(
                reasoning_effort="medium",
                model_connection_path="ignored",
            ).run_episode("Play the game.", output_dir=directory)
            gateway.assert_called_once()

        config_input = next(
            input_text
            for command, input_text in calls
            if "config" in command and "patch" in command
        )
        config = json.loads(config_input)
        self.assertNotIn("env", config)
        self.assertTrue(observed_api_keys)
        self.assertTrue(all(
            key == "openrouter-test-key" for key in observed_api_keys
        ))
        self.assertTrue(
            config["models"]["providers"]["openrouter"]["baseUrl"].startswith(
                "http://127.0.0.1:"
            )
        )
        self.assertEqual(
            config["plugins"],
            {
                "enabled": True,
                "allow": ["openrouter", "duckduckgo", "browser"],
            },
        )
        self.assertEqual(
            config["browser"],
            {"enabled": True, "headless": True, "noSandbox": True,
             "executablePath": "/usr/bin/chromium", "defaultProfile": "openclaw"},
        )
        self.assertEqual(config["gateway"]["bind"], "loopback")
        self.assertEqual(config["gateway"]["auth"], {
            "mode": "token", "token": "${OPENCLAW_GATEWAY_TOKEN}",
        })
        self.assertEqual(config["agents"]["defaults"]["heartbeat"]["every"], "0m")
        plugin_command = next(command for command, _ in calls if "plugins" in command)
        self.assertIn("--link", plugin_command)
        self.assertIn("--accept-capabilities", plugin_command)
        self.assertEqual(plugin_command[-1], directory)
        self.assertEqual(
            config["tools"]["web"]["search"],
            {"enabled": True, "provider": "duckduckgo"},
        )
        self.assertEqual(
            config["tools"]["exec"],
            {"security": "full", "ask": "off"},
        )
        agent_command = next(
            command for command, _input_text in calls if "agent" in command
        )
        self.assertNotIn("--local", agent_command)
        self.assertTrue(
            agent_command[agent_command.index("--session-key") + 1].startswith(
                "agent:main:game-session-"
            )
        )
        self.assertEqual(
            agent_command[agent_command.index("--thinking") + 1],
            "medium",
        )
        self.assertTrue(result.success)

    def test_openclaw_gateway_waits_for_readiness_and_stops_on_agent_error(self):
        for agent_fails in (False, True):
            with self.subTest(agent_fails=agent_fails), tempfile.TemporaryDirectory() as directory:
                process = MagicMock()
                process.poll.return_value = None
                with patch(
                    "clemagents.adapters.openclaw.subprocess.Popen", return_value=process,
                ) as popen, patch(
                    "clemagents.adapters.openclaw.urlopen",
                ) as health:
                    health.return_value.__enter__.return_value.status = 200
                    try:
                        with _openclaw_gateway(
                            ["openclaw", "--profile", "isolated"], Path(directory) / "gateway.txt",
                        ):
                            health.assert_called_once()
                            process.terminate.assert_not_called()
                            if agent_fails:
                                raise ValueError("simulated agent error")
                    except ValueError:
                        self.assertTrue(agent_fails)
                    self.assertEqual(
                        popen.call_args.args[0],
                        ["openclaw", "--profile", "isolated", "gateway", "run"],
                    )
                    process.terminate.assert_called_once()
                    process.wait.assert_called_once_with(timeout=10)

    def test_openclaw_gateway_startup_failure_does_not_run_agent(self):
        with tempfile.TemporaryDirectory() as directory:
            process = MagicMock(returncode=1)
            process.poll.return_value = 1
            with patch(
                "clemagents.adapters.openclaw.subprocess.Popen", return_value=process,
            ), self.assertRaisesRegex(RuntimeError, "Gateway exited during startup"):
                with _openclaw_gateway(["openclaw"], Path(directory) / "gateway.txt"):
                    self.fail("agent must not start without the Gateway")

    def test_openclaw_gateway_startup_timeout_kills_unresponsive_service(self):
        with tempfile.TemporaryDirectory() as directory:
            process = MagicMock()
            process.poll.return_value = None
            process.wait.side_effect = [subprocess.TimeoutExpired("gateway", 10), None]
            with patch(
                "clemagents.adapters.openclaw.subprocess.Popen", return_value=process,
            ), self.assertRaisesRegex(RuntimeError, "Gateway was not ready"):
                with _openclaw_gateway(["openclaw"], Path(directory) / "gateway.txt", startup_timeout=0):
                    self.fail("agent must not start without the Gateway")
            process.terminate.assert_called_once()
            process.kill.assert_called_once()

    def test_openclaw_missing_plugin_fails_before_starting_any_cli(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "clemagents.adapters.openclaw.load_model_connection", return_value=None,
        ), patch(
            "clemagents.adapters.openclaw.DUCKDUCKGO_PLUGIN_PATH", Path(directory) / "missing",
        ), patch("clemagents.adapters.openclaw.subprocess.run") as run:
            result = OpenClawHarness(model="test-model").run_episode("test", output_dir=directory)
            self.assertFalse(result.success)
            self.assertIn("rebuild", result.metadata["runtime_error"])
            run.assert_not_called()

    def test_result_run_dir_orders_openclaw_and_environment_players(self):
        with tempfile.TemporaryDirectory() as directory:
            registry_path = Path(directory) / "agent_registry.json"
            registry_path.write_text(
                json.dumps([
                    {
                        "agent_name": "openclaw-with-gpt-5-mini-openrouter",
                        "backend": "openclaw",
                        "agent_config": {
                            "clem_model": "gpt-5-mini-openrouter"
                        },
                    }
                ]),
                encoding="utf-8",
            )

            run_dir = result_run_dir_name(
                agent_name="openclaw-with-gpt-5-mini-openrouter",
                registry_path=registry_path,
                learner_agent="player_0",
                env_agents={"player_1": "Llama-4-Maverick"},
            )

        self.assertEqual(
            run_dir,
            "openclaw-with-gpt-5-mini-openrouter--Llama-4-Maverick",
        )

    def test_failed_trace_never_reuses_another_harness_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            results_dir = Path(directory)
            wrong_episode = (
                results_dir
                / "hermes-with-gpt-5-mini-openrouter--Llama-4-Maverick"
                / "taboo"
                / "high_en"
                / "episode_00001"
            )
            wrong_episode.mkdir(parents=True)
            (wrong_episode / "instance.json").write_text(
                json.dumps({"game_id": 0}), encoding="utf-8"
            )

            run_dir = (
                "openclaw-with-gpt-5-mini-openrouter--Llama-4-Maverick"
            )
            trace_path = write_agent_artifacts(
                results_dir=str(results_dir),
                run_dir=run_dir,
                game_name="taboo",
                experiment_name="high_en",
                game_id=0,
                before_instance_dirs=set(),
                trace_text="openclaw failure",
                metadata={
                    "started_at": datetime.now(timezone.utc).isoformat(),
                },
            )

            self.assertEqual(
                trace_path.read_text(encoding="utf-8"), "openclaw failure"
            )
            self.assertTrue(
                trace_path.is_relative_to(
                    results_dir / "_agent_failures" / run_dir
                )
            )
            self.assertFalse((wrong_episode / "agent_trace.log").exists())


if __name__ == "__main__":
    unittest.main()
