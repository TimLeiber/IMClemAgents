import base64
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import requests
import yaml
from fastmcp import FastMCP
from fastmcp.tools.base import ToolResult
from mcp.types import ImageContent, TextContent


DEFAULT_OPENENV_MCP_URL = "http://127.0.0.1:8001/mcp"

SERVER_CONFIG_PATH = Path(__file__).parent / "mcp_server_config.yaml"

# response submitted for an agent that stopped before the game was finished
CONTROL_FAILURE_RESPONSE = "AGENT_CONTROL_ERROR: external harness ended before completing the game"
REPEATED_START_MESSAGE = "The episode was aborted because start_game was called more than once."
MODEL_MESSAGE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "message": {"type": "string"},
    },
    "required": ["message"],
    "additionalProperties": False,
}


def _write_completion_marker(result: dict[str, Any], control_failure: bool = False) -> None:
    completion_path = os.environ.get("GAME_COMPLETION_PATH")

    if not completion_path:
        return

    payload = {
        "done": result.get("done") is True,
        "control_failure": control_failure,
        "reward": result.get("reward"),
        "metadata": result.get("metadata") or {},
    }
    path = Path(completion_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


IMAGE_FILE_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
}


def _materialize_observation_images(images: list[dict[str, str]],
                                    observation_dir: str | Path,
                                    observation_index: int) -> list[Path]:
    """Write one observation's transported images into the episode workspace."""
    target_dir = Path(observation_dir) / f"observation_{observation_index:03d}"
    target_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for image_index, image in enumerate(images):
        data = image["data"]
        mime_type = image["mimeType"]
        extension = IMAGE_FILE_EXTENSIONS.get(mime_type)

        if extension is None:
            subtype = mime_type.removeprefix("image/").split("+", 1)[0]
            extension = f".{subtype}" if subtype.isalnum() else ".img"

        image_bytes = base64.b64decode(data, validate=True)
        path = target_dir / f"image_{image_index:02d}{extension}"
        path.write_bytes(image_bytes)
        paths.append(path)

    return paths


def _model_facing_message(result: dict[str, Any],
                          *,
                          observation_dir: str | Path | None = None,
                          observation_index: int = 0) -> ToolResult:
    """Return the environment message as standard MCP content blocks.

    The explicit text block intentionally matches FastMCP's existing
    serialization of ``{"message": ...}`` so text-only games retain their
    current model-facing representation. Images are added as typed MCP image
    blocks rather than serialized into that text.
    """

    context = result.get("context")

    if not isinstance(context, dict):
        context = {}

    content = context.get("content")
    message = {"message": "" if content is None else str(content)}
    blocks = [
        TextContent(
            type="text",
            text=json.dumps(message, ensure_ascii=False, separators=(",", ":")),
        )
    ]

    images = context.get("image") or []

    if not isinstance(images, (list, tuple)):
        images = [images]

    encoded_images = []

    for image in images:
        if not isinstance(image, dict):
            raise ValueError("Model-facing images must be encoded dictionaries")

        data = image.get("data")
        mime_type = image.get("mimeType")

        if not isinstance(data, str) or not isinstance(mime_type, str):
            raise ValueError("Model-facing images require string data and mimeType fields")

        if not mime_type.startswith("image/"):
            raise ValueError(f"Unsupported model-facing media type: {mime_type!r}")

        # Validate the payload before it is passed to a harness or written to
        # the episode workspace.
        base64.b64decode(data, validate=True)
        encoded_images.append({"data": data, "mimeType": mime_type})

        blocks.append(
            ImageContent(
                type="image",
                data=data,
                mimeType=mime_type,
            )
        )

    if encoded_images and observation_dir is not None:
        image_paths = _materialize_observation_images(
            encoded_images,
            observation_dir,
            observation_index,
        )

        paths_text = "\n".join(f"- {path}" for path in image_paths)
        blocks.append(
            TextContent(
                type="text",
                text=(
                    "The observation images are also available as local files:\n"
                    f"{paths_text}"
                ),
            )
        )

    return ToolResult(content=blocks, structured_content=message)


class OpenEnvMCPClient:
    """Client for the session-based JSON-RPC endpoint of the host MCP server.

    This is not the standard MCP client used by external agent harnesses. It
    talks to the OpenEnv /mcp endpoint of the host process, which keeps one
    session per episode, and is used by the bridge below to forward tool calls.
    """

    def __init__(self, mcp_url: str):
        self.mcp_url = mcp_url
        self.session_id: Optional[str] = None
        self._request_id = 0

    def _request(self,
                 method: str,
                 params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send one JSON-RPC request to the host MCP server

        Args:
            method: name of the JSON-RPC method to invoke
            params: optional parameters for the invoked method

        Returns:
            The result payload of the response
        """
        self._request_id += 1

        payload = {"jsonrpc": "2.0",
                   "id": self._request_id,
                   "method": method,
                   "params": params or {}}

        response = requests.post(self.mcp_url,
                                 json=payload,
                                 timeout=(30, None))
        response.raise_for_status()

        data = response.json()

        if "error" in data:
            raise RuntimeError(data["error"])

        return data["result"]

    def create_session(self) -> str:
        """Open an episode session on the host, reusing an already open one

        Returns:
            The identifier of the open session
        """
        if self.session_id is not None:
            return self.session_id

        result = self._request("openenv/session/create")
        self.session_id = result["session_id"]

        # publish the session id through the directory shared with the host
        session_path = os.environ.get("GAME_SESSION_PATH")

        if session_path:
            Path(session_path).write_text(json.dumps({"session_id": self.session_id}),
                                          encoding="utf-8")

        return self.session_id

    def close_session(self) -> None:
        """Close the open episode session on the host"""
        if self.session_id is None:
            return

        self._request("openenv/session/close", {"session_id": self.session_id})
        self.session_id = None

        # The shared file is a recovery marker for a session that is still
        # active when the container exits. Remove it only after the host has
        # confirmed the close; if closing fails, leave it in place so the
        # pipeline can retry cleanup outside the container.
        session_path = os.environ.get("GAME_SESSION_PATH")

        if session_path:
            try:
                Path(session_path).unlink(missing_ok=True)
            except OSError as error:
                print(
                    f"failed to remove closed OpenEnv session marker: {error}",
                    file=sys.stderr,
                )

    def call_tool(self,
                  name: str,
                  arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Invoke one game tool on the host within the episode session

        Args:
            name: name of the tool to invoke
            arguments: optional arguments for the invoked tool

        Returns:
            The data payload returned by the tool
        """
        if name != "start_game" and self.session_id is None:
            raise RuntimeError("No active OpenEnv session. Call start_game first.")

        session_id = self.create_session()

        arguments = arguments or {}

        if name == "submit_response" and arguments.get("response") == CONTROL_FAILURE_RESPONSE:
            name = "abort_game"
            arguments = {"reason": CONTROL_FAILURE_RESPONSE}

        result = self._request("tools/call",
                               {"session_id": session_id,
                                "name": name,
                                "arguments": arguments})

        return result["data"]


def _load_server_instructions() -> str:
    """Read the agent-facing tool instructions from the bridge configuration

    Returns:
        The instructions presented to the external agent
    """
    with open(SERVER_CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config["instructions"].strip()


def create_mcp_bridge(openenv_mcp_url: str | None = None) -> FastMCP:
    """Create the container-side MCP server forwarding game actions to the host.

    External agent harnesses connect to this bridge rather than to the host
    endpoint directly. The bridge exposes the game actions as standard
    MCP tools and translates every call into a session-based request against the
    host MCP server.

    Args:
        openenv_mcp_url: address of the host MCP endpoint, taken from the
            environment when omitted

    Returns:
        The configured MCP server
    """
    openenv_mcp_url = openenv_mcp_url or os.environ.get("OPENENV_MCP_URL", DEFAULT_OPENENV_MCP_URL)

    mcp = FastMCP("game", instructions=_load_server_instructions())
    client = OpenEnvMCPClient(openenv_mcp_url)

    game_state = {"started": False,
                  "done": False,
                  "result": None,
                  "observation_index": 0}
    observation_dir = os.environ.get("GAME_OBSERVATION_DIR")

    def model_facing_message(result: dict[str, Any]) -> ToolResult:
        """Render one sequential observation for the connected harness."""
        observation_index = game_state["observation_index"]
        game_state["observation_index"] += 1
        return _model_facing_message(
            result,
            observation_dir=observation_dir,
            observation_index=observation_index,
        )

    @mcp.tool(output_schema=MODEL_MESSAGE_OUTPUT_SCHEMA)
    def start_game() -> ToolResult:
        """Start the selected game and return its initial environment message."""
        if game_state["started"]:
            if game_state["done"]:
                result = {
                    "context": {
                        "role": "user",
                        "content": REPEATED_START_MESSAGE,
                    },
                    "reward": None,
                    "done": True,
                    "metadata": {"repeated_start_game": True},
                }
                game_state["result"] = result
                _write_completion_marker(result, control_failure=True)
                return model_facing_message(result)

            aborted_result = client.call_tool(
                "submit_response",
                {"response": CONTROL_FAILURE_RESPONSE}
            )
            result = {
                **aborted_result,
                "context": {
                    "role": "user",
                    "content": REPEATED_START_MESSAGE,
                },
                "done": True,
                "metadata": {
                    **(aborted_result.get("metadata") or {}),
                    "repeated_start_game": True,
                },
            }
            game_state["result"] = result
            game_state["done"] = True

            try:
                client.close_session()
            finally:
                _write_completion_marker(result, control_failure=True)

            return model_facing_message(result)

        arguments = {}
        game_id = os.environ.get("GAME_INSTANCE_ID")
        experiment_name = os.environ.get("GAME_EXPERIMENT")

        if game_id is not None:
            arguments["game_id"] = int(game_id)

        if experiment_name is not None:
            arguments["experiment_name"] = experiment_name

        result = client.call_tool("start_game", arguments)

        started_path = os.environ.get("GAME_STARTED_PATH")

        if started_path:
            path = Path(started_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({
                    "session_id": client.session_id,
                    "experiment_name": experiment_name,
                    "game_id": int(game_id) if game_id is not None else None
                }),
                encoding="utf-8"
            )

        game_state["started"] = True
        game_state["result"] = result
        game_state["done"] = result.get("done") is True

        if game_state["done"]:
            try:
                client.close_session()
            finally:
                _write_completion_marker(result)

        return model_facing_message(result)

    @mcp.tool(output_schema=MODEL_MESSAGE_OUTPUT_SCHEMA)
    def submit_response(response: str) -> ToolResult:
        """Submit a response or move to the current game."""
        if not game_state["started"]:
            result = {
                "context": {
                    "role": "user",
                    "content": "The episode was aborted because submit_response was called before start_game.",
                },
                "reward": None,
                "done": True,
                "metadata": {"start_game_required": True},
            }
            game_state["done"] = True
            game_state["result"] = result
            _write_completion_marker(result, control_failure=True)
            return model_facing_message(result)

        if game_state["done"]:
            return model_facing_message(game_state["result"])

        result = client.call_tool("submit_response", {"response": response})
        game_state["result"] = result

        # release the host session as soon as the game reports completion
        if result.get("done") is True:
            game_state["done"] = True

            try:
                client.close_session()
            finally:
                _write_completion_marker(result)

        return model_facing_message(result)
    return mcp


if __name__ == "__main__":
    # container-side entry point started by the agent adapters
    create_mcp_bridge().run()
