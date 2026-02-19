# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.executor.attack import (
    AttackAdversarialConfig,
    AttackConverterConfig,
    AttackParameters,
    AttackScoringConfig,
    ConversationSession,
    ImageEditAttack,
    IEASystemPromptPaths,
    MultiTurnAttackContext,
)
from pyrit.identifiers import ScorerIdentifier, TargetIdentifier
from pyrit.models import (
    AttackOutcome,
    AttackResult,
    Message,
    MessagePiece,
    Score,
    SeedPrompt,
)
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target import PromptChatTarget, PromptTarget
from pyrit.score import FloatScaleScorer, TrueFalseScorer


def _mock_scorer_id(name: str = "MockScorer") -> ScorerIdentifier:
    """Helper to create ScorerIdentifier for tests."""
    return ScorerIdentifier(
        class_name=name,
        class_module="test_module",
        class_description="",
        identifier_type="instance",
    )


def _mock_target_id(name: str = "MockTarget") -> TargetIdentifier:
    """Helper to create TargetIdentifier for tests."""
    return TargetIdentifier(
        class_name=name,
        class_module="test_module",
        class_description="",
        identifier_type="instance",
    )


def _make_image_response(image_path: str = "/images/test.png") -> Message:
    """Helper to create a mock image response."""
    return Message(
        message_pieces=[
            MessagePiece(
                role="assistant",
                original_value=image_path,
                original_value_data_type="image_path",
                converted_value=image_path,
                converted_value_data_type="image_path",
            )
        ]
    )


def _make_text_response(text: str = "Test response") -> Message:
    """Helper to create a mock text response."""
    return Message(
        message_pieces=[
            MessagePiece(
                role="assistant",
                original_value=text,
                original_value_data_type="text",
                converted_value=text,
                converted_value_data_type="text",
            )
        ]
    )


def _make_blocked_response() -> Message:
    """Helper to create a blocked response."""
    return Message(
        message_pieces=[
            MessagePiece(
                role="assistant",
                original_value="blocked",
                original_value_data_type="error",
                converted_value="blocked",
                converted_value_data_type="error",
                response_error="blocked",
            )
        ]
    )


def _make_critique_score(
    score_value: str = "0.6",
    rationale: str = "Image is partially correct",
    correct: str = "person, background",
    missing: str = "ID number, holographic seal",
    wrong: str = "wrong font",
    suggestions: str = "add ID number; add holographic seal",
) -> Score:
    """Helper to create a visual critique score."""
    return Score(
        score_type="float_scale",
        score_value=score_value,
        score_category=["visual_critique"],
        score_value_description="Visual critique score",
        score_rationale=rationale,
        score_metadata={
            "correct_elements": correct,
            "missing_elements": missing,
            "wrong_elements": wrong,
            "improvement_suggestions": suggestions,
        },
        message_piece_id=str(uuid.uuid4()),
        scorer_class_identifier=_mock_scorer_id("CritiqueScorer"),
    )


def _make_objective_score(*, success: bool) -> Score:
    """Helper to create an objective (true/false) score."""
    return Score(
        score_type="true_false",
        score_value="true" if success else "false",
        score_category=["test"],
        score_value_description="Objective score",
        score_rationale="Objective rationale",
        score_metadata={},
        message_piece_id=str(uuid.uuid4()),
        scorer_class_identifier=_mock_scorer_id("ObjectiveScorer"),
    )


@pytest.fixture
def mock_objective_target() -> MagicMock:
    target = MagicMock(spec=PromptTarget)
    target.send_prompt_async = AsyncMock()
    target.get_identifier.return_value = _mock_target_id("MockImageTarget")
    return target


@pytest.fixture
def mock_adversarial_chat() -> MagicMock:
    chat = MagicMock(spec=PromptChatTarget)
    chat.send_prompt_async = AsyncMock()
    chat.set_system_prompt = MagicMock()
    chat.get_identifier.return_value = _mock_target_id("MockChatTarget")
    return chat


@pytest.fixture
def mock_critique_scorer() -> MagicMock:
    scorer = MagicMock(spec=FloatScaleScorer)
    scorer.score_async = AsyncMock(return_value=[_make_critique_score()])
    scorer.get_identifier.return_value = _mock_scorer_id("CritiqueScorer")
    return scorer


@pytest.fixture
def mock_objective_scorer() -> MagicMock:
    scorer = MagicMock(spec=TrueFalseScorer)
    scorer.score_async = AsyncMock(return_value=[_make_objective_score(success=False)])
    scorer.get_identifier.return_value = _mock_scorer_id("ObjectiveScorer")
    return scorer


@pytest.fixture
def mock_prompt_normalizer() -> MagicMock:
    normalizer = MagicMock(spec=PromptNormalizer)
    normalizer.send_prompt_async = AsyncMock()
    return normalizer


@pytest.fixture
def basic_context() -> MultiTurnAttackContext:
    return MultiTurnAttackContext(
        params=AttackParameters(objective="Generate a test image"),
        session=ConversationSession(),
    )


# ============================================================================
# Initialization Tests
# ============================================================================


@pytest.mark.usefixtures("patch_central_database")
class TestImageEditAttackInit:
    """Tests for ImageEditAttack initialization."""

    def test_init_with_required_params(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_critique_scorer: MagicMock,
    ) -> None:
        """Test initialization with minimal required parameters."""
        attack = ImageEditAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=AttackAdversarialConfig(target=mock_adversarial_chat),
            visual_critique_scorer=mock_critique_scorer,
        )

        assert attack._objective_target == mock_objective_target
        assert attack._adversarial_chat == mock_adversarial_chat
        assert attack._visual_critique_scorer == mock_critique_scorer
        assert attack._objective_scorer is None
        assert attack._max_turns == 10
        assert attack._use_edit_mode is True
        assert isinstance(attack._prompt_normalizer, PromptNormalizer)

    def test_init_with_objective_scorer(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_critique_scorer: MagicMock,
        mock_objective_scorer: MagicMock,
    ) -> None:
        """Test initialization with both critique and objective scorers."""
        scoring_config = AttackScoringConfig(objective_scorer=mock_objective_scorer)

        attack = ImageEditAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=AttackAdversarialConfig(target=mock_adversarial_chat),
            visual_critique_scorer=mock_critique_scorer,
            attack_scoring_config=scoring_config,
        )

        assert attack._objective_scorer == mock_objective_scorer

    def test_init_with_custom_max_turns(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_critique_scorer: MagicMock,
    ) -> None:
        """Test initialization with custom max_turns."""
        attack = ImageEditAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=AttackAdversarialConfig(target=mock_adversarial_chat),
            visual_critique_scorer=mock_critique_scorer,
            max_turns=5,
        )

        assert attack._max_turns == 5

    def test_init_raises_on_zero_max_turns(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_critique_scorer: MagicMock,
    ) -> None:
        """Test that zero max_turns raises ValueError."""
        with pytest.raises(ValueError, match="Maximum turns must be a positive integer"):
            ImageEditAttack(
                objective_target=mock_objective_target,
                attack_adversarial_config=AttackAdversarialConfig(target=mock_adversarial_chat),
                visual_critique_scorer=mock_critique_scorer,
                max_turns=0,
            )

    def test_init_with_edit_mode_disabled(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_critique_scorer: MagicMock,
    ) -> None:
        """Test initialization with edit mode disabled."""
        attack = ImageEditAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=AttackAdversarialConfig(target=mock_adversarial_chat),
            visual_critique_scorer=mock_critique_scorer,
            use_edit_mode=False,
        )

        assert attack._use_edit_mode is False

    def test_init_with_custom_system_prompt(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_critique_scorer: MagicMock,
    ) -> None:
        """Test initialization with the IEA system prompt path."""
        attack = ImageEditAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=AttackAdversarialConfig(
                target=mock_adversarial_chat,
                system_prompt_path=IEASystemPromptPaths.IMAGE_EDIT.value,
            ),
            visual_critique_scorer=mock_critique_scorer,
        )

        assert attack._adversarial_chat_system_prompt_template is not None
        assert "objective" in attack._adversarial_chat_system_prompt_template.parameters

    def test_init_with_seed_prompt_string(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_critique_scorer: MagicMock,
    ) -> None:
        """Test initialization with string seed prompt."""
        attack = ImageEditAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=AttackAdversarialConfig(
                target=mock_adversarial_chat,
                seed_prompt="Generate: {{ objective }}",
            ),
            visual_critique_scorer=mock_critique_scorer,
        )

        assert attack._seed_prompt.value == "Generate: {{ objective }}"


# ============================================================================
# Context Validation Tests
# ============================================================================


@pytest.mark.usefixtures("patch_central_database")
class TestContextValidation:
    """Tests for context validation."""

    def test_validate_raises_without_objective(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_critique_scorer: MagicMock,
    ) -> None:
        """Test that validation fails when objective is missing."""
        attack = ImageEditAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=AttackAdversarialConfig(target=mock_adversarial_chat),
            visual_critique_scorer=mock_critique_scorer,
        )

        context = MultiTurnAttackContext(params=AttackParameters(objective=""))
        with pytest.raises(ValueError, match="Attack objective must be provided"):
            attack._validate_context(context=context)

    def test_validate_raises_when_turns_exceeded(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_critique_scorer: MagicMock,
    ) -> None:
        """Test that validation fails when max turns already exceeded."""
        attack = ImageEditAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=AttackAdversarialConfig(target=mock_adversarial_chat),
            visual_critique_scorer=mock_critique_scorer,
            max_turns=3,
        )

        context = MultiTurnAttackContext(
            params=AttackParameters(objective="Test"),
        )
        context.executed_turns = 3
        with pytest.raises(ValueError, match="Already exceeded max turns"):
            attack._validate_context(context=context)

    def test_validate_passes_with_valid_context(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_critique_scorer: MagicMock,
        basic_context: MultiTurnAttackContext,
    ) -> None:
        """Test that validation passes with valid context."""
        attack = ImageEditAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=AttackAdversarialConfig(target=mock_adversarial_chat),
            visual_critique_scorer=mock_critique_scorer,
        )

        # Should not raise
        attack._validate_context(context=basic_context)


# ============================================================================
# Target Message Building Tests
# ============================================================================


@pytest.mark.usefixtures("patch_central_database")
class TestTargetMessageBuilding:
    """Tests for building messages to the image target."""

    def test_text_only_on_first_turn(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_critique_scorer: MagicMock,
    ) -> None:
        """Test that first turn sends text-only message (generate mode)."""
        attack = ImageEditAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=AttackAdversarialConfig(target=mock_adversarial_chat),
            visual_critique_scorer=mock_critique_scorer,
        )

        message = attack._build_target_message(text_prompt="Generate a cat")
        assert len(message.message_pieces) == 1
        assert message.message_pieces[0].converted_value_data_type == "text"
        assert message.message_pieces[0].original_value == "Generate a cat"

    def test_text_plus_image_on_subsequent_turn(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_critique_scorer: MagicMock,
    ) -> None:
        """Test that subsequent turns send text+image message (edit mode)."""
        attack = ImageEditAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=AttackAdversarialConfig(target=mock_adversarial_chat),
            visual_critique_scorer=mock_critique_scorer,
        )
        attack._last_image_path = "/images/prev.png"

        message = attack._build_target_message(text_prompt="Add a hat to the cat")
        assert len(message.message_pieces) == 2
        assert message.message_pieces[0].converted_value_data_type == "text"
        assert message.message_pieces[0].original_value == "Add a hat to the cat"
        assert message.message_pieces[1].converted_value_data_type == "image_path"
        assert message.message_pieces[1].original_value == "/images/prev.png"

    def test_text_only_when_edit_mode_disabled(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_critique_scorer: MagicMock,
    ) -> None:
        """Test that edit mode disabled sends text-only even with prior image."""
        attack = ImageEditAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=AttackAdversarialConfig(target=mock_adversarial_chat),
            visual_critique_scorer=mock_critique_scorer,
            use_edit_mode=False,
        )
        attack._last_image_path = "/images/prev.png"

        message = attack._build_target_message(text_prompt="Generate a cat")
        assert len(message.message_pieces) == 1
        assert message.message_pieces[0].converted_value_data_type == "text"

    def test_shared_conversation_id_in_edit_message(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_critique_scorer: MagicMock,
    ) -> None:
        """Test that text and image pieces share the same conversation_id."""
        attack = ImageEditAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=AttackAdversarialConfig(target=mock_adversarial_chat),
            visual_critique_scorer=mock_critique_scorer,
        )
        attack._last_image_path = "/images/prev.png"

        message = attack._build_target_message(text_prompt="Edit this")
        conv_ids = {p.conversation_id for p in message.message_pieces}
        assert len(conv_ids) == 1, "All pieces should share the same conversation_id"


# ============================================================================
# Adversarial Chat Message Building Tests
# ============================================================================


@pytest.mark.usefixtures("patch_central_database")
class TestAdversarialChatMessage:
    """Tests for building messages to the adversarial chat."""

    def test_seed_prompt_on_first_turn(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_critique_scorer: MagicMock,
        basic_context: MultiTurnAttackContext,
    ) -> None:
        """Test that first turn uses seed prompt."""
        attack = ImageEditAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=AttackAdversarialConfig(target=mock_adversarial_chat),
            visual_critique_scorer=mock_critique_scorer,
        )

        message = attack._build_adversarial_chat_message(context=basic_context)
        assert len(message.message_pieces) == 1
        assert message.message_pieces[0].converted_value_data_type == "text"
        # Should contain the objective if the seed prompt has a template
        assert "Generate a test image" in message.get_value()

    def test_blocked_response_returns_fallback(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_critique_scorer: MagicMock,
        basic_context: MultiTurnAttackContext,
    ) -> None:
        """Test that blocked response returns the default fallback message."""
        attack = ImageEditAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=AttackAdversarialConfig(target=mock_adversarial_chat),
            visual_critique_scorer=mock_critique_scorer,
        )
        basic_context.last_response = _make_blocked_response()

        message = attack._build_adversarial_chat_message(context=basic_context)
        assert "blocked by content filters" in message.get_value()

    def test_image_response_builds_multimodal_message(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_critique_scorer: MagicMock,
        basic_context: MultiTurnAttackContext,
    ) -> None:
        """Test that image response produces multimodal message with critique + image."""
        attack = ImageEditAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=AttackAdversarialConfig(target=mock_adversarial_chat),
            visual_critique_scorer=mock_critique_scorer,
        )
        basic_context.last_response = _make_image_response("/images/gen.png")
        basic_context.last_score = _make_critique_score()

        message = attack._build_adversarial_chat_message(context=basic_context)
        assert len(message.message_pieces) == 2
        text_piece = message.message_pieces[0]
        image_piece = message.message_pieces[1]
        assert text_piece.converted_value_data_type == "text"
        assert image_piece.converted_value_data_type == "image_path"
        assert image_piece.original_value == "/images/gen.png"

    def test_text_response_sends_critique_only(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_critique_scorer: MagicMock,
        basic_context: MultiTurnAttackContext,
    ) -> None:
        """Test that non-image response sends text-only critique feedback."""
        attack = ImageEditAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=AttackAdversarialConfig(target=mock_adversarial_chat),
            visual_critique_scorer=mock_critique_scorer,
        )
        basic_context.last_response = _make_text_response("Some text result")
        basic_context.last_score = _make_critique_score()

        message = attack._build_adversarial_chat_message(context=basic_context)
        assert len(message.message_pieces) == 1
        assert message.message_pieces[0].converted_value_data_type == "text"


# ============================================================================
# Critique Feedback Tests
# ============================================================================


@pytest.mark.usefixtures("patch_central_database")
class TestCritiqueFeedback:
    """Tests for building critique feedback strings."""

    def test_formats_all_metadata_fields(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_critique_scorer: MagicMock,
    ) -> None:
        """Test that all metadata fields are included in feedback."""
        attack = ImageEditAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=AttackAdversarialConfig(target=mock_adversarial_chat),
            visual_critique_scorer=mock_critique_scorer,
        )

        score = _make_critique_score(
            correct="person",
            missing="hat",
            wrong="color",
            suggestions="add hat; fix color",
        )
        feedback = attack._build_critique_feedback(score=score)

        assert "0.6" in feedback
        assert "Correct: person" in feedback
        assert "Missing: hat" in feedback
        assert "Wrong: color" in feedback
        assert "Suggestions: add hat; fix color" in feedback

    def test_handles_none_score(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_critique_scorer: MagicMock,
    ) -> None:
        """Test that None score returns fallback message."""
        attack = ImageEditAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=AttackAdversarialConfig(target=mock_adversarial_chat),
            visual_critique_scorer=mock_critique_scorer,
        )

        feedback = attack._build_critique_feedback(score=None)
        assert "No critique feedback" in feedback

    def test_handles_score_without_metadata(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_critique_scorer: MagicMock,
    ) -> None:
        """Test feedback with a score that has no metadata."""
        attack = ImageEditAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=AttackAdversarialConfig(target=mock_adversarial_chat),
            visual_critique_scorer=mock_critique_scorer,
        )

        score = Score(
            score_type="float_scale",
            score_value="0.5",
            score_category=["test"],
            score_value_description="Test",
            score_rationale="Looks incomplete",
            score_metadata={},
            message_piece_id=str(uuid.uuid4()),
            scorer_class_identifier=_mock_scorer_id(),
        )
        feedback = attack._build_critique_feedback(score=score)

        assert "0.5" in feedback
        assert "Looks incomplete" in feedback
        # Should not have metadata fields
        assert "Correct:" not in feedback
        assert "Missing:" not in feedback


# ============================================================================
# Image Tracking Tests
# ============================================================================


@pytest.mark.usefixtures("patch_central_database")
class TestImageTracking:
    """Tests for tracking the last generated image."""

    def test_updates_image_path_from_response(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_critique_scorer: MagicMock,
    ) -> None:
        """Test that _update_last_image extracts image path."""
        attack = ImageEditAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=AttackAdversarialConfig(target=mock_adversarial_chat),
            visual_critique_scorer=mock_critique_scorer,
        )

        response = _make_image_response("/images/new.png")
        attack._update_last_image(response=response)
        assert attack._last_image_path == "/images/new.png"

    def test_does_not_update_on_error(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_critique_scorer: MagicMock,
    ) -> None:
        """Test that error responses don't update the image path."""
        attack = ImageEditAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=AttackAdversarialConfig(target=mock_adversarial_chat),
            visual_critique_scorer=mock_critique_scorer,
        )
        attack._last_image_path = "/images/old.png"

        attack._update_last_image(response=_make_blocked_response())
        assert attack._last_image_path == "/images/old.png"

    def test_does_not_update_on_none(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_critique_scorer: MagicMock,
    ) -> None:
        """Test that None response doesn't crash."""
        attack = ImageEditAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=AttackAdversarialConfig(target=mock_adversarial_chat),
            visual_critique_scorer=mock_critique_scorer,
        )

        attack._update_last_image(response=None)
        assert attack._last_image_path is None

    @pytest.mark.asyncio
    async def test_teardown_clears_image_path(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_critique_scorer: MagicMock,
        basic_context: MultiTurnAttackContext,
    ) -> None:
        """Test that teardown resets image tracking state."""
        attack = ImageEditAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=AttackAdversarialConfig(target=mock_adversarial_chat),
            visual_critique_scorer=mock_critique_scorer,
        )
        attack._last_image_path = "/images/leftover.png"

        await attack._teardown_async(context=basic_context)
        assert attack._last_image_path is None


# ============================================================================
# Success Check Tests
# ============================================================================


@pytest.mark.usefixtures("patch_central_database")
class TestSuccessCheck:
    """Tests for the _check_success logic."""

    def test_objective_scorer_true_means_success(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_critique_scorer: MagicMock,
    ) -> None:
        """Test that objective scorer returning true means success."""
        attack = ImageEditAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=AttackAdversarialConfig(target=mock_adversarial_chat),
            visual_critique_scorer=mock_critique_scorer,
        )

        result = attack._check_success(
            critique_score=_make_critique_score(score_value="0.5"),
            objective_score=_make_objective_score(success=True),
        )
        assert result is True

    def test_objective_scorer_false_means_failure(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_critique_scorer: MagicMock,
    ) -> None:
        """Test that objective scorer returning false means failure."""
        attack = ImageEditAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=AttackAdversarialConfig(target=mock_adversarial_chat),
            visual_critique_scorer=mock_critique_scorer,
        )

        result = attack._check_success(
            critique_score=_make_critique_score(score_value="0.95"),
            objective_score=_make_objective_score(success=False),
        )
        assert result is False

    def test_critique_above_threshold_without_objective_scorer(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_critique_scorer: MagicMock,
    ) -> None:
        """Test that critique score above threshold succeeds when no objective scorer."""
        attack = ImageEditAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=AttackAdversarialConfig(target=mock_adversarial_chat),
            visual_critique_scorer=mock_critique_scorer,
        )

        result = attack._check_success(
            critique_score=_make_critique_score(score_value="0.95"),
            objective_score=None,
        )
        assert result is True

    def test_critique_below_threshold_without_objective_scorer(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_critique_scorer: MagicMock,
    ) -> None:
        """Test that critique score below threshold fails when no objective scorer."""
        attack = ImageEditAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=AttackAdversarialConfig(target=mock_adversarial_chat),
            visual_critique_scorer=mock_critique_scorer,
        )

        result = attack._check_success(
            critique_score=_make_critique_score(score_value="0.5"),
            objective_score=None,
        )
        assert result is False

    def test_no_scores_means_failure(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_critique_scorer: MagicMock,
    ) -> None:
        """Test that no scores means failure."""
        attack = ImageEditAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=AttackAdversarialConfig(target=mock_adversarial_chat),
            visual_critique_scorer=mock_critique_scorer,
        )

        result = attack._check_success(critique_score=None, objective_score=None)
        assert result is False

    def test_objective_scorer_takes_precedence(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_critique_scorer: MagicMock,
    ) -> None:
        """Test that objective scorer result takes precedence over critique threshold."""
        attack = ImageEditAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=AttackAdversarialConfig(target=mock_adversarial_chat),
            visual_critique_scorer=mock_critique_scorer,
        )

        # Critique says success (0.95 >= 0.9 threshold) but objective says failure
        result = attack._check_success(
            critique_score=_make_critique_score(score_value="0.95"),
            objective_score=_make_objective_score(success=False),
        )
        assert result is False


# ============================================================================
# Scoring Tests
# ============================================================================


@pytest.mark.usefixtures("patch_central_database")
class TestScoring:
    """Tests for scoring methods."""

    @pytest.mark.asyncio
    async def test_critique_scorer_called_with_response(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_critique_scorer: MagicMock,
        basic_context: MultiTurnAttackContext,
    ) -> None:
        """Test that critique scorer is called with the response message."""
        attack = ImageEditAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=AttackAdversarialConfig(target=mock_adversarial_chat),
            visual_critique_scorer=mock_critique_scorer,
        )
        basic_context.last_response = _make_image_response()

        score = await attack._score_with_critique_async(context=basic_context)

        mock_critique_scorer.score_async.assert_called_once()
        assert score is not None
        assert score.score_type == "float_scale"

    @pytest.mark.asyncio
    async def test_critique_scorer_returns_none_without_response(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_critique_scorer: MagicMock,
        basic_context: MultiTurnAttackContext,
    ) -> None:
        """Test that critique scorer returns None when no response."""
        attack = ImageEditAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=AttackAdversarialConfig(target=mock_adversarial_chat),
            visual_critique_scorer=mock_critique_scorer,
        )

        score = await attack._score_with_critique_async(context=basic_context)
        assert score is None

    @pytest.mark.asyncio
    async def test_objective_scorer_returns_none_when_not_configured(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_critique_scorer: MagicMock,
        basic_context: MultiTurnAttackContext,
    ) -> None:
        """Test that objective scoring returns None when no objective scorer."""
        attack = ImageEditAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=AttackAdversarialConfig(target=mock_adversarial_chat),
            visual_critique_scorer=mock_critique_scorer,
        )
        basic_context.last_response = _make_image_response()

        score = await attack._score_objective_async(context=basic_context)
        assert score is None


# ============================================================================
# Attack Loop Tests
# ============================================================================


@pytest.mark.usefixtures("patch_central_database")
class TestPerformLoop:
    """Tests for the main attack execution loop."""

    @pytest.mark.asyncio
    async def test_single_turn_failure(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_critique_scorer: MagicMock,
        mock_prompt_normalizer: MagicMock,
        basic_context: MultiTurnAttackContext,
    ) -> None:
        """Test single-turn attack that fails."""
        # Adversarial chat returns text prompt
        mock_prompt_normalizer.send_prompt_async.return_value = _make_text_response("Draw a cat")

        attack = ImageEditAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=AttackAdversarialConfig(target=mock_adversarial_chat),
            visual_critique_scorer=mock_critique_scorer,
            prompt_normalizer=mock_prompt_normalizer,
            max_turns=1,
        )

        # Mock internal methods for clean test
        with patch.object(attack, "_send_to_target_async", new_callable=AsyncMock) as mock_send, \
             patch.object(attack, "_score_with_critique_async", new_callable=AsyncMock) as mock_crit, \
             patch.object(attack, "_score_objective_async", new_callable=AsyncMock) as mock_obj:
            mock_send.return_value = _make_image_response()
            mock_crit.return_value = _make_critique_score(score_value="0.3")
            mock_obj.return_value = None

            result = await attack._perform_async(context=basic_context)

        assert result.outcome == AttackOutcome.FAILURE
        assert result.executed_turns == 1

    @pytest.mark.asyncio
    async def test_success_on_first_turn(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_critique_scorer: MagicMock,
        mock_prompt_normalizer: MagicMock,
        basic_context: MultiTurnAttackContext,
    ) -> None:
        """Test that attack succeeds immediately when objective is met on first turn."""
        mock_prompt_normalizer.send_prompt_async.return_value = _make_text_response("Draw a cat")

        attack = ImageEditAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=AttackAdversarialConfig(target=mock_adversarial_chat),
            visual_critique_scorer=mock_critique_scorer,
            prompt_normalizer=mock_prompt_normalizer,
            max_turns=5,
        )

        with patch.object(attack, "_send_to_target_async", new_callable=AsyncMock) as mock_send, \
             patch.object(attack, "_score_with_critique_async", new_callable=AsyncMock) as mock_crit, \
             patch.object(attack, "_score_objective_async", new_callable=AsyncMock) as mock_obj:
            mock_send.return_value = _make_image_response()
            mock_crit.return_value = _make_critique_score(score_value="0.95")
            mock_obj.return_value = _make_objective_score(success=True)

            result = await attack._perform_async(context=basic_context)

        assert result.outcome == AttackOutcome.SUCCESS
        assert result.executed_turns == 1

    @pytest.mark.asyncio
    async def test_multi_turn_edit_chaining(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_critique_scorer: MagicMock,
        mock_prompt_normalizer: MagicMock,
        basic_context: MultiTurnAttackContext,
    ) -> None:
        """Test that image path is tracked across turns for edit mode."""
        mock_prompt_normalizer.send_prompt_async.return_value = _make_text_response("Draw a cat")

        attack = ImageEditAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=AttackAdversarialConfig(target=mock_adversarial_chat),
            visual_critique_scorer=mock_critique_scorer,
            prompt_normalizer=mock_prompt_normalizer,
            max_turns=2,
        )

        call_count = 0

        async def mock_send(*, context, message):
            nonlocal call_count
            call_count += 1
            return _make_image_response(f"/images/gen_{call_count}.png")

        with patch.object(attack, "_send_to_target_async", side_effect=mock_send), \
             patch.object(attack, "_score_with_critique_async", new_callable=AsyncMock) as mock_crit, \
             patch.object(attack, "_score_objective_async", new_callable=AsyncMock) as mock_obj:
            mock_crit.return_value = _make_critique_score(score_value="0.3")
            mock_obj.return_value = None

            result = await attack._perform_async(context=basic_context)

        assert result.executed_turns == 2
        assert attack._last_image_path == "/images/gen_2.png"

    @pytest.mark.asyncio
    async def test_stops_early_on_success(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_critique_scorer: MagicMock,
        mock_prompt_normalizer: MagicMock,
        basic_context: MultiTurnAttackContext,
    ) -> None:
        """Test that attack stops early when objective is achieved."""
        mock_prompt_normalizer.send_prompt_async.return_value = _make_text_response("Draw a cat")

        attack = ImageEditAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=AttackAdversarialConfig(target=mock_adversarial_chat),
            visual_critique_scorer=mock_critique_scorer,
            prompt_normalizer=mock_prompt_normalizer,
            max_turns=10,
        )

        turn = 0

        async def mock_obj_score(*, context):
            nonlocal turn
            turn += 1
            # Succeed on turn 3
            if turn >= 3:
                return _make_objective_score(success=True)
            return _make_objective_score(success=False)

        with patch.object(attack, "_send_to_target_async", new_callable=AsyncMock) as mock_send, \
             patch.object(attack, "_score_with_critique_async", new_callable=AsyncMock) as mock_crit, \
             patch.object(attack, "_score_objective_async", side_effect=mock_obj_score):
            mock_send.return_value = _make_image_response()
            mock_crit.return_value = _make_critique_score(score_value="0.5")

            result = await attack._perform_async(context=basic_context)

        assert result.outcome == AttackOutcome.SUCCESS
        assert result.executed_turns == 3


# ============================================================================
# System Prompt Path Tests
# ============================================================================


@pytest.mark.usefixtures("patch_central_database")
class TestSystemPromptPaths:
    """Tests for IEASystemPromptPaths enum."""

    def test_image_edit_path_exists(self) -> None:
        """Test that the image edit system prompt file exists."""
        assert IEASystemPromptPaths.IMAGE_EDIT.value.exists()

    def test_image_edit_path_is_yaml(self) -> None:
        """Test that the system prompt path is a YAML file."""
        assert str(IEASystemPromptPaths.IMAGE_EDIT.value).endswith(".yaml")
