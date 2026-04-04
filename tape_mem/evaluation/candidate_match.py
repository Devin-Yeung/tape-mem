from collections.abc import Sequence

from tape_mem.types import EvaluationResult, Evaluator


class CandidateMatchEvaluator(Evaluator):
    """Deterministic evaluator that only normalizes whitespace and casing."""

    def evaluate(
        self,
        prediction: str,
        candidate_answers: Sequence[str],
    ) -> EvaluationResult:
        normalized_prediction = _normalize_text(prediction)
        normalized_candidates = tuple(
            _normalize_text(candidate_answer) for candidate_answer in candidate_answers
        )

        return EvaluationResult(
            matched=normalized_prediction in normalized_candidates,
            normalized_prediction=normalized_prediction,
            normalized_candidates=normalized_candidates,
        )


def _normalize_text(value: str) -> str:
    return " ".join(value.split()).casefold()
