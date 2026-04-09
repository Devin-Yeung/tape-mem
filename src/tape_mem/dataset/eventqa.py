from mashumaro.mixins.json import DataClassJSONMixin
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal, TypeAlias, cast, Dict

from datasets import load_dataset

# EventQA v0 only supports the three upstream Accurate Retrieval variants that
# share the same context-injection and multi-question execution model.
EventQAVariant: TypeAlias = Literal[
    "eventqa_full",
    "eventqa_65536",
    "eventqa_131072",
]

SUPPORTED_EVENTQA_VARIANTS: tuple[EventQAVariant, ...] = (
    "eventqa_full",
    "eventqa_65536",
    "eventqa_131072",
)

DatasetLoader: TypeAlias = Callable[[str], Any]


class EventQADatasetError(ValueError):
    """Raised when the upstream dataset shape is incompatible with the adapter."""


@dataclass(frozen=True)
class EventQAQuestion(DataClassJSONMixin):
    """One aligned EventQA question together with its accepted gold answers."""

    question_id: str
    text: str
    answer_candidates: tuple[str, ...]


@dataclass(frozen=True)
class EventQAExample:
    """One EventQA context row with all questions that share the same memory."""

    example_id: str
    variant: EventQAVariant
    context: str
    questions: tuple[EventQAQuestion, ...]


def naive_eventqa_example() -> EventQAExample:
    """Return a tiny EventQA-shaped fixture for tests and local experiments.

    The upstream dataset uses very long contexts, but most unit tests only need
    one realistic example that exercises the typed adapter contract without
    depending on Hugging Face downloads.
    """

    return EventQAExample(
        example_id="eventqa_full:naive",
        variant="eventqa_full",
        context=(
            "On March 14, 2024, Lena Park organized the Harbor Book Club meeting "
            "at the Seaview Library. She brought a blue notebook to track the "
            "group's reading list and announced that the next discussion would "
            "cover the novel 'Northline' on April 2, 2024."
        ),
        questions=(
            EventQAQuestion(
                question_id="naive-q1",
                text="Who organized the Harbor Book Club meeting?",
                answer_candidates=("Lena Park",),
            ),
            EventQAQuestion(
                question_id="naive-q2",
                text="Where was the Harbor Book Club meeting held?",
                answer_candidates=("Seaview Library", "the Seaview Library"),
            ),
            EventQAQuestion(
                question_id="naive-q3",
                text="What color notebook did Lena Park bring?",
                answer_candidates=("blue", "a blue notebook", "blue notebook"),
            ),
            EventQAQuestion(
                question_id="naive-q4",
                text="When is the next discussion scheduled?",
                answer_candidates=("April 2, 2024",),
            ),
        ),
    )


def load_eventqa_examples(
    *,
    dataset_name: str = "ai-hyz/MemoryAgentBench",
    split_name: str = "Accurate_Retrieval",
) -> list[EventQAExample]:
    """Load and validate the EventQA rows used by the v0 benchmark.

    The adapter isolates all Hugging Face row-shape knowledge in one place so
    the runner can operate on typed records instead of untyped dictionaries.
    """

    dataset = load_dataset(dataset_name)

    try:
        rows = cast(Iterable[dict[str, Any]], dataset[split_name])
    except Exception as exc:  # pragma: no cover - defensive boundary around HF
        raise EventQADatasetError(
            f"Unable to access split {split_name!r} from dataset {dataset_name!r}"
        ) from exc

    examples: list[EventQAExample] = []

    counter: Dict[str, int] = {}

    for row_index, row in enumerate(rows):
        variant = _read_variant(row)
        if variant is None:
            continue

        counter[variant] = counter.get(variant, -1) + 1

        questions = _build_questions(row)
        context = _require_string(row, "context")

        examples.append(
            EventQAExample(
                example_id=f"{variant}_{counter[variant]}",
                variant=variant,
                context=context,
                questions=questions,
            )
        )

    return examples


def _read_variant(row: dict[str, Any]) -> EventQAVariant | None:
    metadata = _require_mapping(row, "metadata")
    source = metadata.get("source")

    if not isinstance(source, str):
        raise EventQADatasetError(
            "EventQA rows must include metadata.source as a string"
        )

    if source not in SUPPORTED_EVENTQA_VARIANTS:
        return None

    return cast(EventQAVariant, source)


def _build_questions(row: dict[str, Any]) -> tuple[EventQAQuestion, ...]:
    questions = _require_string_list(row, "questions")
    answers = _require_answer_list(row, "answers")
    metadata = _require_mapping(row, "metadata")
    qa_pair_ids = _require_string_list(metadata, "qa_pair_ids")

    if not (len(questions) == len(answers) == len(qa_pair_ids)):
        raise EventQADatasetError(
            "EventQA rows must have aligned lengths across questions, answers, and metadata.qa_pair_ids"
        )

    return tuple(
        EventQAQuestion(
            question_id=question_id,
            text=question,
            answer_candidates=tuple(answer_candidates),
        )
        for question, answer_candidates, question_id in zip(
            questions, answers, qa_pair_ids, strict=True
        )
    )


def _require_mapping(container: dict[str, Any], field_name: str) -> dict[str, Any]:
    value = container.get(field_name)
    if not isinstance(value, dict):
        raise EventQADatasetError(f"Expected {field_name} to be a mapping")
    return value


def _require_string(container: dict[str, Any], field_name: str) -> str:
    value = container.get(field_name)
    if not isinstance(value, str):
        raise EventQADatasetError(f"Expected {field_name} to be a string")
    return value


def _require_string_list(container: dict[str, Any], field_name: str) -> list[str]:
    value = container.get(field_name)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise EventQADatasetError(f"Expected {field_name} to be a list[str]")
    return value


def _require_answer_list(container: dict[str, Any], field_name: str) -> list[list[str]]:
    value = container.get(field_name)
    if not isinstance(value, list):
        raise EventQADatasetError(f"Expected {field_name} to be a list[list[str]]")

    if not all(
        isinstance(answer_candidates, list)
        and all(isinstance(candidate, str) for candidate in answer_candidates)
        for answer_candidates in value
    ):
        raise EventQADatasetError(f"Expected {field_name} to be a list[list[str]]")

    return value
