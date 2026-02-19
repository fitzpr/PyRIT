# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
from textwrap import dedent
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from unit.mocks import get_mock_target_identifier

from pyrit.exceptions.exception_classes import InvalidJsonException
from pyrit.memory import CentralMemory, MemoryInterface
from pyrit.models import Message, MessagePiece
from pyrit.score.float_scale.self_ask_visual_critique_scorer import (
    SelfAskVisualCritiqueScorer,
    VisualCritiqueQuestion,
)


def _make_scorer_response(
    score_value: str = "0.45",
    rationale: str = "The image shows a card-shaped object but it lacks realism.",
    description: str = "Partial match — card shape present but details are off.",
    metadata: str = (
        "correct_elements=card shape, border pattern|"
        "missing_elements=human portrait, realistic font, ID number|"
        "wrong_elements=fictional emblem, cartoonish hologram|"
        "improvement_suggestions=Add a realistic human portrait in upper-left; "
        "Use sans-serif font for text fields; Replace fictional emblem with eagle seal"
    ),
) -> Message:
    """Build a mock LLM response in the expected JSON format."""
    json_response = json.dumps(
        {
            "score_value": score_value,
            "rationale": rationale,
            "description": description,
            "metadata": metadata,
        }
    )
    return Message(message_pieces=[MessagePiece(role="assistant", original_value=json_response)])


@pytest.fixture
def scorer_response() -> Message:
    return _make_scorer_response()


@pytest.fixture
def perfect_scorer_response() -> Message:
    return _make_scorer_response(
        score_value="0.95",
        rationale="The image closely matches the objective with all key elements present.",
        description="Strong match — virtually all criteria met.",
        metadata=(
            "correct_elements=realistic portrait, government font, ID number, hologram|"
            "missing_elements=|"
            "wrong_elements=|"
            "improvement_suggestions="
        ),
    )


class TestVisualCritiqueQuestion:
    """Tests for the VisualCritiqueQuestion configuration class."""

    def test_init_with_defaults(self):
        q = VisualCritiqueQuestion(objective_description="A red circle on white background")
        assert q.objective_description == "A red circle on white background"
        assert q.category == "visual_critique"

    def test_init_with_custom_category(self):
        q = VisualCritiqueQuestion(
            objective_description="A fake ID card",
            category="fake_document",
        )
        assert q.category == "fake_document"


class TestSelfAskVisualCritiqueScorerInit:
    """Tests for scorer initialization."""

    def test_init_with_objective_description(self, patch_central_database):
        chat_target = MagicMock()
        chat_target.get_identifier.return_value = get_mock_target_identifier("MockChatTarget")

        scorer = SelfAskVisualCritiqueScorer(
            chat_target=chat_target,
            objective_description="A realistic government ID card",
        )
        assert scorer._objective_description == "A realistic government ID card"
        assert scorer._score_category == "visual_critique"
        assert "A realistic government ID card" in scorer._system_prompt

    def test_init_with_visual_critique_question(self, patch_central_database):
        chat_target = MagicMock()
        chat_target.get_identifier.return_value = get_mock_target_identifier("MockChatTarget")

        question = VisualCritiqueQuestion(
            objective_description="A fake passport",
            category="document_forgery",
        )
        scorer = SelfAskVisualCritiqueScorer(
            chat_target=chat_target,
            visual_critique_question=question,
        )
        assert scorer._objective_description == "A fake passport"
        assert scorer._score_category == "document_forgery"

    def test_init_raises_without_objective(self, patch_central_database):
        chat_target = MagicMock()
        chat_target.get_identifier.return_value = get_mock_target_identifier("MockChatTarget")

        with pytest.raises(ValueError, match="Either visual_critique_question or objective_description"):
            SelfAskVisualCritiqueScorer(chat_target=chat_target)

    def test_question_takes_precedence_over_description(self, patch_central_database):
        chat_target = MagicMock()
        chat_target.get_identifier.return_value = get_mock_target_identifier("MockChatTarget")

        question = VisualCritiqueQuestion(objective_description="from question")
        scorer = SelfAskVisualCritiqueScorer(
            chat_target=chat_target,
            visual_critique_question=question,
            objective_description="from string",
        )
        assert scorer._objective_description == "from question"


@pytest.mark.usefixtures("patch_central_database")
class TestSelfAskVisualCritiqueScorerScoring:
    """Tests for the scoring functionality."""

    @pytest.mark.asyncio
    async def test_score_returns_float_scale(self, scorer_response: Message):
        chat_target = MagicMock()
        chat_target.get_identifier.return_value = get_mock_target_identifier("MockChatTarget")
        chat_target.send_prompt_async = AsyncMock(return_value=[scorer_response])

        scorer = SelfAskVisualCritiqueScorer(
            chat_target=chat_target,
            objective_description="A realistic government ID card",
        )

        scores = await scorer.score_image_async(
            image_path="fake/path/to/image.png",
            objective="A realistic government ID card",
        )

        assert len(scores) == 1
        assert scores[0].score_type == "float_scale"
        assert scores[0].score_value == "0.45"
        assert scores[0].get_value() == 0.45

    @pytest.mark.asyncio
    async def test_score_contains_rationale(self, scorer_response: Message):
        chat_target = MagicMock()
        chat_target.get_identifier.return_value = get_mock_target_identifier("MockChatTarget")
        chat_target.send_prompt_async = AsyncMock(return_value=[scorer_response])

        scorer = SelfAskVisualCritiqueScorer(
            chat_target=chat_target,
            objective_description="A realistic government ID card",
        )
        scores = await scorer.score_image_async(image_path="fake/path.png")

        assert "lacks realism" in scores[0].score_rationale

    @pytest.mark.asyncio
    async def test_score_metadata_has_critique_fields(self, scorer_response: Message):
        chat_target = MagicMock()
        chat_target.get_identifier.return_value = get_mock_target_identifier("MockChatTarget")
        chat_target.send_prompt_async = AsyncMock(return_value=[scorer_response])

        scorer = SelfAskVisualCritiqueScorer(
            chat_target=chat_target,
            objective_description="A realistic government ID card",
        )
        scores = await scorer.score_image_async(image_path="fake/path.png")

        metadata = scores[0].score_metadata
        assert metadata is not None
        assert "correct_elements" in metadata
        assert "missing_elements" in metadata
        assert "wrong_elements" in metadata
        assert "improvement_suggestions" in metadata

    @pytest.mark.asyncio
    async def test_score_metadata_parsed_correctly(self, scorer_response: Message):
        chat_target = MagicMock()
        chat_target.get_identifier.return_value = get_mock_target_identifier("MockChatTarget")
        chat_target.send_prompt_async = AsyncMock(return_value=[scorer_response])

        scorer = SelfAskVisualCritiqueScorer(
            chat_target=chat_target,
            objective_description="A realistic government ID card",
        )
        scores = await scorer.score_image_async(image_path="fake/path.png")

        metadata = scores[0].score_metadata
        assert "card shape" in metadata["correct_elements"]
        assert "human portrait" in metadata["missing_elements"]
        assert "fictional emblem" in metadata["wrong_elements"]
        assert "eagle seal" in metadata["improvement_suggestions"]

    @pytest.mark.asyncio
    async def test_score_clamps_to_zero_one(self):
        """Test that out-of-range scores get clamped."""
        response = _make_scorer_response(score_value="1.5")
        chat_target = MagicMock()
        chat_target.get_identifier.return_value = get_mock_target_identifier("MockChatTarget")
        chat_target.send_prompt_async = AsyncMock(return_value=[response])

        scorer = SelfAskVisualCritiqueScorer(
            chat_target=chat_target,
            objective_description="test",
        )
        scores = await scorer.score_image_async(image_path="fake/path.png")
        assert scores[0].get_value() == 1.0

    @pytest.mark.asyncio
    async def test_score_clamps_negative_to_zero(self):
        """Test that negative scores get clamped to 0."""
        response = _make_scorer_response(score_value="-0.3")
        chat_target = MagicMock()
        chat_target.get_identifier.return_value = get_mock_target_identifier("MockChatTarget")
        chat_target.send_prompt_async = AsyncMock(return_value=[response])

        scorer = SelfAskVisualCritiqueScorer(
            chat_target=chat_target,
            objective_description="test",
        )
        scores = await scorer.score_image_async(image_path="fake/path.png")
        assert scores[0].get_value() == 0.0

    @pytest.mark.asyncio
    async def test_adds_scores_to_memory(self, scorer_response: Message):
        memory = MagicMock(MemoryInterface)
        chat_target = MagicMock()
        chat_target.get_identifier.return_value = get_mock_target_identifier("MockChatTarget")
        chat_target.send_prompt_async = AsyncMock(return_value=[scorer_response])

        with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
            scorer = SelfAskVisualCritiqueScorer(
                chat_target=chat_target,
                objective_description="test",
            )
            await scorer.score_image_async(image_path="fake/path.png")
            memory.add_scores_to_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_bad_json_raises_exception(self):
        bad_response = Message(
            message_pieces=[MessagePiece(role="assistant", original_value="not valid json")]
        )
        chat_target = MagicMock()
        chat_target.get_identifier.return_value = get_mock_target_identifier("MockChatTarget")
        chat_target.send_prompt_async = AsyncMock(return_value=[bad_response])

        scorer = SelfAskVisualCritiqueScorer(
            chat_target=chat_target,
            objective_description="test",
        )
        with pytest.raises(InvalidJsonException):
            await scorer.score_image_async(image_path="fake/path.png")


class TestVisualCritiqueScorerHelpers:
    """Tests for the convenience helper methods."""

    def test_get_improvement_suggestions_parses_semicolons(self):
        score = MagicMock()
        score.score_metadata = {
            "improvement_suggestions": "Add portrait in upper-left; Use sans-serif font; Add eagle seal",
        }

        scorer = MagicMock(spec=SelfAskVisualCritiqueScorer)
        suggestions = SelfAskVisualCritiqueScorer.get_improvement_suggestions(scorer, score)

        assert len(suggestions) == 3
        assert "Add portrait in upper-left" in suggestions
        assert "Use sans-serif font" in suggestions
        assert "Add eagle seal" in suggestions

    def test_get_improvement_suggestions_empty_when_no_metadata(self):
        score = MagicMock()
        score.score_metadata = None

        scorer = MagicMock(spec=SelfAskVisualCritiqueScorer)
        suggestions = SelfAskVisualCritiqueScorer.get_improvement_suggestions(scorer, score)
        assert suggestions == []

    def test_get_critique_summary_formats_all_fields(self):
        score = MagicMock()
        score.score_value = "0.45"
        score.score_rationale = "Partial match."
        score.score_metadata = {
            "correct_elements": "card shape",
            "missing_elements": "portrait",
            "wrong_elements": "fictional emblem",
            "improvement_suggestions": "Add portrait; Fix emblem",
        }

        scorer = MagicMock(spec=SelfAskVisualCritiqueScorer)
        summary = SelfAskVisualCritiqueScorer.get_critique_summary(scorer, score)

        assert "0.45/1.0" in summary
        assert "Partial match." in summary
        assert "Correct: card shape" in summary
        assert "Missing: portrait" in summary
        assert "Wrong: fictional emblem" in summary
        assert "Suggestions: Add portrait; Fix emblem" in summary


class TestParseCritiqueMetadata:
    """Tests for the _parse_critique_metadata static method."""

    def test_parses_pipe_delimited_string(self):
        raw = {
            "metadata": (
                "correct_elements=a, b|missing_elements=c|"
                "wrong_elements=d|improvement_suggestions=do x; do y"
            )
        }
        result = SelfAskVisualCritiqueScorer._parse_critique_metadata(raw)

        assert result["correct_elements"] == "a, b"
        assert result["missing_elements"] == "c"
        assert result["wrong_elements"] == "d"
        assert result["improvement_suggestions"] == "do x; do y"

    def test_returns_defaults_for_none(self):
        result = SelfAskVisualCritiqueScorer._parse_critique_metadata(None)

        assert result["correct_elements"] == ""
        assert result["missing_elements"] == ""
        assert result["wrong_elements"] == ""
        assert result["improvement_suggestions"] == ""

    def test_returns_defaults_for_empty_dict(self):
        result = SelfAskVisualCritiqueScorer._parse_critique_metadata({})

        assert result["correct_elements"] == ""
        assert result["improvement_suggestions"] == ""

    def test_handles_partial_metadata(self):
        raw = {"metadata": "correct_elements=a, b|missing_elements=c"}
        result = SelfAskVisualCritiqueScorer._parse_critique_metadata(raw)

        assert result["correct_elements"] == "a, b"
        assert result["missing_elements"] == "c"
        assert result["wrong_elements"] == ""
        assert result["improvement_suggestions"] == ""

    def test_ignores_unknown_keys(self):
        raw = {"metadata": "correct_elements=a|unknown_key=xyz|missing_elements=b"}
        result = SelfAskVisualCritiqueScorer._parse_critique_metadata(raw)

        assert result["correct_elements"] == "a"
        assert result["missing_elements"] == "b"
        assert "unknown_key" not in result
