# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
from typing import Optional

from pyrit.common import net_utility
from pyrit.models import Message, construct_response_from_request
from pyrit.prompt_target import PromptTarget, limit_requests_per_minute

logger = logging.getLogger(__name__)


class MoltbotTarget(PromptTarget):
    """
    A prompt target for Moltbot/Clawdbot (OpenClaw) instances.

    Moltbot (formerly Clawdbot) is an open-source, local AI agent that runs autonomously
    and can perform actions across different platforms. This target allows PyRIT to interact
    with and test Moltbot instances via their HTTP API.

    More information: https://github.com/steinbergerbernd/moltbot

    Args:
        endpoint_uri: The base URI of the Moltbot API (e.g., "http://localhost:18789").
        channel: The communication channel to send messages through (e.g., "cli", "telegram", "whatsapp").
                 Defaults to "cli" for command-line interface.
        api_key: Optional API key for authentication if the Moltbot instance requires it.
        max_requests_per_minute: Number of requests the target can handle per minute before
                                  hitting a rate limit. The number of requests sent to the target
                                  will be capped at the value provided.
    """

    def __init__(
        self,
        *,
        endpoint_uri: str = "http://localhost:18789",
        channel: str = "cli",
        api_key: Optional[str] = None,
        max_requests_per_minute: Optional[int] = None,
    ) -> None:
        """
        Initialize the Moltbot target.
        """
        # Ensure endpoint doesn't have trailing slash
        self._base_endpoint = endpoint_uri.rstrip("/")
        self._send_endpoint = f"{self._base_endpoint}/api/send"

        super().__init__(
            max_requests_per_minute=max_requests_per_minute, endpoint=self._send_endpoint, model_name="moltbot"
        )

        self._channel = channel
        self._api_key = api_key

    @limit_requests_per_minute
    async def send_prompt_async(self, *, message: Message) -> list[Message]:
        """
        Send a prompt to the Moltbot instance.

        Args:
            message: The message to send, containing one or more message pieces.

        Returns:
            A list containing the response message from Moltbot.

        Raises:
            ValueError: If the message format is invalid or the response is empty.
        """
        self._validate_request(message=message)
        request = message.message_pieces[0]

        logger.info(f"Sending the following prompt to the Moltbot target: {request}")

        response = await self._send_message_async(request.converted_value)

        response_entry = construct_response_from_request(request=request, response_text_pieces=[response])

        return [response_entry]

    def _validate_request(self, *, message: Message) -> None:
        """
        Validate that the request message is in the correct format.

        Args:
            message: The message to validate.

        Raises:
            ValueError: If the message has more than one piece or is not text.
        """
        n_pieces = len(message.message_pieces)
        if n_pieces != 1:
            raise ValueError(f"This target only supports a single message piece. Received: {n_pieces} pieces.")

        piece_type = message.message_pieces[0].converted_value_data_type
        if piece_type != "text":
            raise ValueError(f"This target only supports text prompt input. Received: {piece_type}.")

    async def _send_message_async(self, text: str) -> str:
        """
        Send a message to the Moltbot API and return the response.

        Args:
            text: The message text to send.

        Returns:
            The response text from Moltbot.

        Raises:
            ValueError: If the response is empty or invalid.
        """
        payload: dict[str, object] = {
            "channel": self._channel,
            "message": text,
        }

        # Add API key to headers if provided
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        resp = await net_utility.make_request_and_raise_if_error_async(
            endpoint_uri=self._send_endpoint,
            method="POST",
            request_body=payload,
            post_type="json",
            headers=headers if headers else None,
        )

        if not resp.text:
            raise ValueError("The Moltbot API returned an empty response.")

        try:
            json_response = resp.json()
            # Extract the response based on expected API structure
            # The actual response format may vary, so we try multiple common formats
            if isinstance(json_response, dict):
                response_text = (
                    json_response.get("response")
                    or json_response.get("message")
                    or json_response.get("reply")
                    or json_response.get("text")
                    or str(json_response)
                )
            else:
                response_text = str(json_response)
        except json.JSONDecodeError:
            # If response is not JSON, use the raw text
            response_text = resp.text

        logger.info(f'Received the following response from Moltbot: "{response_text}"')
        return response_text
