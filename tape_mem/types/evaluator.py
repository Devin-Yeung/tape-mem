from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from mashumaro.mixins.json import DataClassJSONMixin


@dataclass(frozen=True)
class EvaluationResult(DataClassJSONMixin):
    """Normalized evaluation output that artifact writers can persist directly."""

    matched: bool
    normalized_prediction: str
    normalized_candidates: tuple[str, ...]


@runtime_checkable
class Evaluator(Protocol):
    """A pluggable evaluator for benchmark predictions."""

    def evaluate(
        self,
        prediction: str,
        candidate_answers: Sequence[str],
    ) -> EvaluationResult: ...
