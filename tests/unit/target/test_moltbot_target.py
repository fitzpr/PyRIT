# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.models import Message, MessagePiece
from pyrit.prompt_target import MoltbotTarget
from unit.mocks import get_image_message_piece


@pytest.fixture
def moltbot_target(patch_central_database) -> MoltbotTarget:
    return MoltbotTarget()


def test_moltbot_initializes_with_defaults(moltbot_target: MoltbotTarget):
    assert moltbot_target
    assert moltbot_target._channel == "cli"
    assert moltbot_target._api_key is None


def test_moltbot_initializes_with_custom_endpoint():
    target = MoltbotTarget(endpoint_uri="http://custom-host:8080")
    identifier = target.get_identifier()
    assert identifier["endpoint"] == "http://custom-host:8080/api/send"


def test_moltbot_initializes_with_custom_channel():
    target = MoltbotTarget(channel="telegram")
    assert target._channel == "telegram"


def test_moltbot_initializes_with_api_key():
    target = MoltbotTarget(api_key="test_key_123")
    assert target._api_key == "test_key_123"


def test_moltbot_sets_endpoint_and_rate_limit():
    target = MoltbotTarget(endpoint_uri="http://localhost:18789", max_requests_per_minute=10)
    identifier = target.get_identifier()
    assert identifier["endpoint"] == "http://localhost:18789/api/send"
    assert target._max_requests_per_minute == 10


def test_moltbot_strips_trailing_slash():
    target = MoltbotTarget(endpoint_uri="http://localhost:18789/")
    assert target._base_endpoint == "http://localhost:18789"
    assert target._send_endpoint == "http://localhost:18789/api/send"


@pytest.mark.asyncio
async def test_moltbot_validate_request_length(moltbot_target: MoltbotTarget):
    request = Message(
        message_pieces=[
            MessagePiece(role="user", conversation_id="123", original_value="test"),
            MessagePiece(role="user", conversation_id="123", original_value="test2"),
        ]
    )
    with pytest.raises(ValueError, match="This target only supports a single message piece."):
        await moltbot_target.send_prompt_async(message=request)


@pytest.mark.asyncio
async def test_moltbot_validate_prompt_type(moltbot_target: MoltbotTarget):
    request = Message(message_pieces=[get_image_message_piece()])
    with pytest.raises(ValueError, match="This target only supports text prompt input."):
        await moltbot_target.send_prompt_async(message=request)


@pytest.mark.asyncio
async def test_moltbot_send_prompt_async_success():
    target = MoltbotTarget()

    # Create a mock response
    mock_response = MagicMock()
    mock_response.text = '{"response": "Hello from Moltbot"}'
    mock_response.json.return_value = {"response": "Hello from Moltbot"}

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async",
        new_callable=AsyncMock,
        return_value=mock_response,
    ) as mock_request:
        request = Message(
            message_pieces=[
                MessagePiece(role="user", conversation_id="123", original_value="Hello", converted_value="Hello")
            ]
        )

        result = await target.send_prompt_async(message=request)

        # Verify the request was made correctly
        mock_request.assert_called_once()
        call_kwargs = mock_request.call_args.kwargs
        assert call_kwargs["endpoint_uri"] == "http://localhost:18789/api/send"
        assert call_kwargs["method"] == "POST"
        assert call_kwargs["request_body"]["channel"] == "cli"
        assert call_kwargs["request_body"]["message"] == "Hello"
        assert call_kwargs["post_type"] == "json"

        # Verify the response
        assert len(result) == 1
        assert result[0].message_pieces[0].converted_value == "Hello from Moltbot"


@pytest.mark.asyncio
async def test_moltbot_send_prompt_async_with_api_key():
    target = MoltbotTarget(api_key="test_key_123")

    mock_response = MagicMock()
    mock_response.text = '{"response": "Authenticated response"}'
    mock_response.json.return_value = {"response": "Authenticated response"}

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async",
        new_callable=AsyncMock,
        return_value=mock_response,
    ) as mock_request:
        request = Message(
            message_pieces=[
                MessagePiece(role="user", conversation_id="123", original_value="Test", converted_value="Test")
            ]
        )

        await target.send_prompt_async(message=request)

        # Verify API key was included in headers
        call_kwargs = mock_request.call_args.kwargs
        assert call_kwargs["headers"]["Authorization"] == "Bearer test_key_123"


@pytest.mark.asyncio
async def test_moltbot_send_prompt_handles_different_response_formats():
    target = MoltbotTarget()

    # Test various response formats that might be returned
    test_cases = [
        ('{"response": "test1"}', "test1"),
        ('{"message": "test2"}', "test2"),
        ('{"reply": "test3"}', "test3"),
        ('{"text": "test4"}', "test4"),
        ('{"unknown_key": "test5"}', "test5"),  # Will convert dict to string
        ("Plain text response", "Plain text response"),
    ]

    for response_text, expected_value in test_cases:
        mock_response = MagicMock()
        mock_response.text = response_text

        # Try to parse as JSON, fall back to text
        try:
            mock_response.json.return_value = json.loads(response_text)
        except json.JSONDecodeError:
            mock_response.json.side_effect = json.JSONDecodeError("test", "", 0)

        with patch(
            "pyrit.common.net_utility.make_request_and_raise_if_error_async",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            request = Message(
                message_pieces=[
                    MessagePiece(role="user", conversation_id="123", original_value="Test", converted_value="Test")
                ]
            )

            result = await target.send_prompt_async(message=request)
            assert expected_value in result[0].message_pieces[0].converted_value


@pytest.mark.asyncio
async def test_moltbot_send_prompt_empty_response_raises_error():
    target = MoltbotTarget()

    mock_response = MagicMock()
    mock_response.text = ""

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        request = Message(
            message_pieces=[
                MessagePiece(role="user", conversation_id="123", original_value="Test", converted_value="Test")
            ]
        )

        with pytest.raises(ValueError, match="The Moltbot API returned an empty response."):
            await target.send_prompt_async(message=request)


@pytest.mark.asyncio
async def test_moltbot_send_prompt_with_custom_channel():
    target = MoltbotTarget(channel="telegram")

    mock_response = MagicMock()
    mock_response.text = '{"response": "test"}'
    mock_response.json.return_value = {"response": "test"}

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async",
        new_callable=AsyncMock,
        return_value=mock_response,
    ) as mock_request:
        request = Message(
            message_pieces=[
                MessagePiece(role="user", conversation_id="123", original_value="Test", converted_value="Test")
            ]
        )

        await target.send_prompt_async(message=request)

        # Verify the correct channel was used
        call_kwargs = mock_request.call_args.kwargs
        assert call_kwargs["request_body"]["channel"] == "telegram"
