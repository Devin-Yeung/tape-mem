import ast
import re
from dataclasses import dataclass
from datetime import date, datetime
from datetime import time as dt_time
from typing import Any, Iterable, Literal, TypeAlias, cast

from datasets import load_dataset
from loguru import logger
from mashumaro.mixins.json import DataClassJSONMixin

LongMemEvalVariant: TypeAlias = Literal["longmemeval_s*"]

SUPPORTED_LONGMEMEVAL_VARIANTS: tuple[LongMemEvalVariant, ...] = ("longmemeval_s*",)


class LongMemEvalDatasetError(ValueError):
    """Raised when the upstream dataset shape is incompatible with the adapter."""


@dataclass(frozen=True)
class LongMemEvalMessage(DataClassJSONMixin):
    """One message in a conversation session."""

    role: str
    content: str


@dataclass(frozen=True)
class LongMemEvalSession(DataClassJSONMixin):
    """One chat session identified by a timestamp."""

    # Preserve full datetime (date + time) when available.
    chat_time: datetime
    messages: tuple[LongMemEvalMessage, ...]


@dataclass(frozen=True)
class LongMemEvalQuestion(DataClassJSONMixin):
    """One aligned question together with its accepted gold answers."""

    question_id: str
    question_text: str
    time_info: str
    answer_candidates: tuple[str, ...]


@dataclass(frozen=True)
class LongMemEvalExample:
    """One LongMemEval example with structured conversation sessions and aligned questions."""

    example_id: str
    variant: str
    sessions: tuple[LongMemEvalSession, ...]
    questions: tuple[LongMemEvalQuestion, ...]


def load_longmemeval_examples(
    *,
    dataset_name: str = "ai-hyz/MemoryAgentBench",
    split_name: str = "Accurate_Retrieval",
) -> list[LongMemEvalExample]:
    """Load and validate the LongMemEval rows from HuggingFace.

    Each LongMemEval context is a series of chat sessions, parsed from a string
    representation of a Python list structure: [header1, messages1, header2, messages2, ...]
    """

    dataset = load_dataset(dataset_name)

    try:
        rows = cast(Iterable[dict[str, Any]], dataset[split_name])
    except Exception as exc:  # pragma: no cover - defensive boundary around HF
        raise LongMemEvalDatasetError(
            f"Unable to access split {split_name!r} from dataset {dataset_name!r}"
        ) from exc

    examples: list[LongMemEvalExample] = []

    counter: dict[str, int] = {}

    for row_index, row in enumerate(rows):
        variant = _read_variant(row)
        if variant is None:
            continue

        counter[variant] = counter.get(variant, -1) + 1

        sessions = _build_sessions(row)
        questions = _build_questions(row)

        examples.append(
            LongMemEvalExample(
                example_id=f"{variant}_{counter[variant]}",
                variant=variant,
                sessions=sessions,
                questions=questions,
            )
        )

    return examples


def _read_variant(row: dict[str, Any]) -> str | None:
    metadata = _require_mapping(row, "metadata")
    source = metadata.get("source")

    if not isinstance(source, str):
        raise LongMemEvalDatasetError(
            "LongMemEval rows must include metadata.source as a string"
        )

    if source not in SUPPORTED_LONGMEMEVAL_VARIANTS:
        return None

    return source


def _parse_chat_time(header: str) -> datetime:
    """Parse a chat header string and return a datetime.

    This parser only accepts the canonical form observed in the dataset:
      "Chat Time: 2022/11/17 (Thu) 12:04"

    Both a YYYY/MM/DD date and an H:MM time must be present. If either is
    missing or unparseable, a LongMemEvalDatasetError is raised.
    """
    # Extract a date in YYYY/MM/DD form
    date_match = re.search(r"(\d{4}/\d{1,2}/\d{1,2})", header)
    if not date_match:
        raise LongMemEvalDatasetError(
            f"Expected header to include a date in YYYY/MM/DD form: {header!r}"
        )

    date_str = date_match.group(1)
    try:
        y, m, d = [int(p) for p in date_str.split("/")]
        parsed_date = date(y, m, d)
    except Exception as exc:  # pragma: no cover - defensive
        raise LongMemEvalDatasetError(
            f"Unable to parse date portion {date_str!r} from header {header!r}"
        ) from exc

    # Extract time portion (require HH:MM or H:MM)
    time_match = re.search(r"(\d{1,2}:\d{2})", header)
    if not time_match:
        raise LongMemEvalDatasetError(
            f"Expected header to include a time in H:MM form: {header!r}"
        )

    time_str = time_match.group(1)
    try:
        t = dt_time.fromisoformat(time_str)
    except Exception:
        parts = time_str.split(":")
        try:
            h = int(parts[0])
            minute = int(parts[1])
            t = dt_time(h, minute)
        except Exception as exc:  # pragma: no cover - defensive
            raise LongMemEvalDatasetError(
                f"Unable to parse time portion {time_str!r} from header {header!r}"
            ) from exc

    return datetime.combine(parsed_date, t)


def _build_sessions(row: dict[str, Any]) -> tuple[LongMemEvalSession, ...]:
    """Parse the context string into structured sessions.

    The raw context is a string representation of a Python list:
    ["Chat Time: ...", [{role, content}, ...], "Chat Time: ...", [{role, content}, ...], ...]
    """
    context = _require_string(row, "context")

    parsed = ast.literal_eval(context)

    if not isinstance(parsed, list):
        raise LongMemEvalDatasetError("Context must be a list")

    sessions: list[LongMemEvalSession] = []
    i = 0
    while i < len(parsed):
        if not isinstance(parsed[i], str):
            i += 1
            continue

        header = parsed[i]
        i += 1

        if i >= len(parsed) or not isinstance(parsed[i], list):
            continue

        message_dicts = parsed[i]
        i += 1

        messages: list[LongMemEvalMessage] = []
        for msg_dict in message_dicts:
            if not isinstance(msg_dict, dict):
                continue
            role = msg_dict.get("role", "")
            content = msg_dict.get("content", "")
            if role and content:
                messages.append(LongMemEvalMessage(role=role, content=content))

        if messages:
            try:
                parsed_time = _parse_chat_time(header)
            except LongMemEvalDatasetError:
                # If parsing fails, skip this session as it cannot be reliably mapped
                # to a time value.
                logger.error("Failed to parse chat time: %s", header)
                continue

            sessions.append(
                LongMemEvalSession(chat_time=parsed_time, messages=tuple(messages))
            )

    return tuple(sessions)


def _build_questions(row: dict[str, Any]) -> tuple[LongMemEvalQuestion, ...]:
    questions = _require_string_list(row, "questions")
    answers = _require_answer_list(row, "answers")
    metadata = _require_mapping(row, "metadata")
    qa_pair_ids = _require_string_list(metadata, "qa_pair_ids")

    if not (len(questions) == len(answers) == len(qa_pair_ids)):
        raise LongMemEvalDatasetError(
            "LongMemEval rows must have aligned lengths across questions, answers, and metadata.qa_pair_ids"
        )

    return tuple(
        LongMemEvalQuestion(
            question_id=question_id,
            question_text=_extract_question_text(question),
            time_info=_extract_time_info(question),
            answer_candidates=tuple(answer_candidates),
        )
        for question, answer_candidates, question_id in zip(
            questions, answers, qa_pair_ids, strict=True
        )
    )


_TIME_INFO_PATTERN = "Current Date: "
_QUESTION_PREFIX = "Now Answer the Question:"


def _extract_time_info(text: str) -> str:
    """Extract the time information from a question text."""
    start = text.find(_TIME_INFO_PATTERN)
    if start == -1:
        return ""
    end = text.find(",", start)
    if end == -1:
        return ""
    return text[start:end].strip()


def _extract_question_text(text: str) -> str:
    """Extract the actual question text (without time info prefix)."""
    idx = text.find(_QUESTION_PREFIX)
    if idx == -1:
        return text.strip()
    return text[idx:].strip()


def _require_mapping(container: dict[str, Any], field_name: str) -> dict[str, Any]:
    value = container.get(field_name)
    if not isinstance(value, dict):
        raise LongMemEvalDatasetError(f"Expected {field_name} to be a mapping")
    return value


def _require_string(container: dict[str, Any], field_name: str) -> str:
    value = container.get(field_name)
    if not isinstance(value, str):
        raise LongMemEvalDatasetError(f"Expected {field_name} to be a string")
    return value


def _require_string_list(container: dict[str, Any], field_name: str) -> list[str]:
    value = container.get(field_name)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise LongMemEvalDatasetError(f"Expected {field_name} to be a list[str]")
    return value


def _require_answer_list(container: dict[str, Any], field_name: str) -> list[list[str]]:
    value = container.get(field_name)
    if not isinstance(value, list):
        raise LongMemEvalDatasetError(f"Expected {field_name} to be a list[list[str]]")

    if not all(
        isinstance(answer_candidates, list)
        and all(isinstance(candidate, str) for candidate in answer_candidates)
        for answer_candidates in value
    ):
        raise LongMemEvalDatasetError(f"Expected {field_name} to be a list[list[str]]")

    return value
