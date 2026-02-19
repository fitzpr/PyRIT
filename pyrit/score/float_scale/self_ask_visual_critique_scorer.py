# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path
from typing import Optional, Union

from pyrit.common import verify_and_resolve_path
from pyrit.common.path import DATASETS_PATH
from pyrit.identifiers import ScorerIdentifier
from pyrit.models import MessagePiece, Score, SeedPrompt
from pyrit.prompt_target import PromptChatTarget
from pyrit.score.float_scale.float_scale_scorer import FloatScaleScorer
from pyrit.score.scorer_prompt_validator import ScorerPromptValidator

VISUAL_CRITIQUE_PATH = Path(DATASETS_PATH, "score", "visual_critique").resolve()


class VisualCritiqueQuestion:
    """
    Defines an objective for visual critique scoring.

    This describes what the image SHOULD depict, so the scorer can evaluate
    how closely a generated image matches the intended objective.
    """

    def __init__(self, *, objective_description: str, category: str = "visual_critique"):
        """
        Initialize a VisualCritiqueQuestion.

        Args:
            objective_description: A detailed description of what the image should depict.
                The scorer will evaluate how closely the image matches this description.
            category: The score category label. Defaults to "visual_critique".
        """
        self.objective_description = objective_description
        self.category = category


class SelfAskVisualCritiqueScorer(FloatScaleScorer):
    """
    A scorer that provides detailed visual critique of generated images.

    Instead of a binary True/False, this scorer returns:
    - A float score (0.0-1.0) indicating how closely the image matches the objective
    - Structured critique identifying correct, missing, and wrong visual elements
    - Actionable improvement suggestions that can be fed back to an image generator

    This scorer is designed for the image edit/remix attack pipeline where
    granular visual feedback enables iterative image refinement.

    The critique details are stored in the score's metadata field as a dict with keys:
    - "correct_elements": Visual elements that match the objective
    - "missing_elements": Visual elements that are absent but required
    - "wrong_elements": Visual elements that are present but incorrect
    - "improvement_suggestions": Specific prompt refinements for the next iteration
    """

    _DEFAULT_VALIDATOR: ScorerPromptValidator = ScorerPromptValidator(
        supported_data_types=["image_path"],
    )

    def __init__(
        self,
        *,
        chat_target: PromptChatTarget,
        visual_critique_question: Optional[VisualCritiqueQuestion] = None,
        objective_description: Optional[str] = None,
        system_prompt_path: Optional[Union[str, Path]] = None,
        validator: Optional[ScorerPromptValidator] = None,
    ) -> None:
        """
        Initialize the SelfAskVisualCritiqueScorer.

        Args:
            chat_target: The multimodal chat target (e.g. GPT-4o) used for scoring.
                Must support image inputs.
            visual_critique_question: A VisualCritiqueQuestion defining the objective.
            objective_description: Alternative to visual_critique_question — just the description string.
                If both are provided, visual_critique_question takes precedence.
            system_prompt_path: Custom path to the system prompt YAML template.
                Defaults to the built-in visual_critique_system_prompt.yaml.
            validator: Custom validator. Defaults to image_path-only validation.

        Raises:
            ValueError: If neither visual_critique_question nor objective_description is provided.
        """
        super().__init__(validator=validator or self._DEFAULT_VALIDATOR)

        self._prompt_target = chat_target

        if visual_critique_question is None and objective_description is None:
            raise ValueError("Either visual_critique_question or objective_description must be provided.")

        if visual_critique_question is None:
            visual_critique_question = VisualCritiqueQuestion(objective_description=objective_description)

        self._score_category = visual_critique_question.category
        self._objective_description = visual_critique_question.objective_description

        system_prompt_path = (
            system_prompt_path
            if system_prompt_path
            else VISUAL_CRITIQUE_PATH / "visual_critique_system_prompt.yaml"
        )
        system_prompt_path = verify_and_resolve_path(system_prompt_path)

        scoring_instructions_template = SeedPrompt.from_yaml_file(system_prompt_path)
        self._system_prompt = scoring_instructions_template.render_template_value(
            objective_description=self._objective_description,
        )

    def _build_identifier(self) -> ScorerIdentifier:
        """Build the scorer evaluation identifier."""
        return self._create_identifier(
            system_prompt_template=self._system_prompt,
            user_prompt_template="objective: {objective}\nimage: {image}",
            prompt_target=self._prompt_target,
        )

    async def _score_piece_async(self, message_piece: MessagePiece, *, objective: Optional[str] = None) -> list[Score]:
        """
        Score an image piece with detailed visual critique.

        Args:
            message_piece: The message piece containing the image to score.
            objective: Optional runtime objective override. If not provided,
                uses the objective_description from initialization.

        Returns:
            A list containing a single Score with:
            - score_value: float 0.0-1.0 as string
            - score_rationale: Step-by-step analysis
            - score_metadata: Dict with correct_elements, missing_elements,
              wrong_elements, and improvement_suggestions
        """
        effective_objective = objective or self._objective_description

        # Image content is sent directly; objective is prepended as text context
        prepended_text = f"objective: {effective_objective}\nimage:"
        scoring_value = message_piece.converted_value
        scoring_data_type = message_piece.converted_value_data_type

        unvalidated_score = await self._score_value_with_llm(
            prompt_target=self._prompt_target,
            system_prompt=self._system_prompt,
            message_value=scoring_value,
            message_data_type=scoring_data_type,
            scored_prompt_id=message_piece.id,
            category=self._score_category,
            objective=effective_objective,
            attack_identifier=message_piece.attack_identifier,
        )

        # Parse the float score value and clamp to [0, 1]
        raw_float = float(unvalidated_score.raw_score_value)
        clamped = max(0.0, min(1.0, raw_float))

        score = unvalidated_score.to_score(
            score_value=str(clamped),
            score_type="float_scale",
        )

        # Parse the structured critique from the metadata string.
        # The LLM returns metadata as a pipe-delimited string:
        #   "correct_elements=...|missing_elements=...|wrong_elements=...|improvement_suggestions=..."
        score.score_metadata = self._parse_critique_metadata(score.score_metadata)

        return [score]

    @staticmethod
    def _parse_critique_metadata(
        raw_metadata: Optional[dict[str, Union[str, int, float]]],
    ) -> dict[str, str]:
        """
        Parse the structured critique from the metadata field.

        The LLM returns metadata as a pipe-delimited string packed into the "metadata" key:
            "correct_elements=...|missing_elements=...|wrong_elements=...|improvement_suggestions=..."

        This method unpacks it into a proper dict.

        Args:
            raw_metadata: The raw metadata dict from the score (may contain a single string value).

        Returns:
            A dict with keys: correct_elements, missing_elements, wrong_elements, improvement_suggestions.
        """
        result: dict[str, str] = {
            "correct_elements": "",
            "missing_elements": "",
            "wrong_elements": "",
            "improvement_suggestions": "",
        }

        if not raw_metadata:
            return result

        # The metadata may come as {"metadata": "correct_elements=...|..."} or as the string directly
        raw_str = ""
        if isinstance(raw_metadata, dict):
            # metadata_output_key default is "metadata", so the parsed value is a string
            raw_str = str(raw_metadata.get("metadata", ""))
            if not raw_str:
                # Maybe the LLM put the structured string as the only value
                raw_str = str(next(iter(raw_metadata.values()), ""))
        elif isinstance(raw_metadata, str):
            raw_str = raw_metadata

        if not raw_str:
            return result

        # Parse pipe-delimited key=value pairs
        for segment in raw_str.split("|"):
            segment = segment.strip()
            if "=" in segment:
                key, _, value = segment.partition("=")
                key = key.strip()
                if key in result:
                    result[key] = value.strip()

        return result

    def get_improvement_suggestions(self, score: Score) -> list[str]:
        """
        Extract actionable improvement suggestions from a score.

        This is a convenience method for the attack pipeline to get
        the list of prompt refinements from a score's metadata.

        Args:
            score: A Score object returned by this scorer.

        Returns:
            A list of improvement suggestion strings. Empty list if none available.
        """
        if score.score_metadata and "improvement_suggestions" in score.score_metadata:
            suggestions_str = str(score.score_metadata["improvement_suggestions"])
            return [s.strip() for s in suggestions_str.split(";") if s.strip()]
        return []

    def get_critique_summary(self, score: Score) -> str:
        """
        Build a formatted critique string suitable for feeding back to an adversarial LLM.

        Args:
            score: A Score object returned by this scorer.

        Returns:
            A formatted string with the score, what's correct, what's missing/wrong,
            and specific suggestions for the next prompt.
        """
        metadata = score.score_metadata or {}
        parts = [
            f"Score: {score.score_value}/1.0",
            f"Analysis: {score.score_rationale}",
        ]

        if metadata.get("correct_elements"):
            parts.append(f"Correct: {metadata['correct_elements']}")
        if metadata.get("missing_elements"):
            parts.append(f"Missing: {metadata['missing_elements']}")
        if metadata.get("wrong_elements"):
            parts.append(f"Wrong: {metadata['wrong_elements']}")
        if metadata.get("improvement_suggestions"):
            parts.append(f"Suggestions: {metadata['improvement_suggestions']}")

        return "\n".join(parts)
