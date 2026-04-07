from dataclasses import dataclass
from typing import Generic, List, TypeVar
from mashumaro.mixins.json import DataClassJSONMixin

from .agent import AgentResponse
from ..dataset import EventQAQuestion

T = TypeVar("T")


@dataclass(frozen=True)
class QueryResult(Generic[T], DataClassJSONMixin):
    """One query and the corresponding agent response.

    Generic experiment records stay reusable across benchmarks, but callers that
    need JSON serialization should concretize the type parameter in a subclass so
    mashumaro can generate a serializer for the actual payload type.
    """

    question: T
    response: AgentResponse


@dataclass(frozen=True)
class Experiment(Generic[T], DataClassJSONMixin):
    results: List[QueryResult[T]]


@dataclass(frozen=True)
class EventQAQueryResult(QueryResult[EventQAQuestion]):
    """Concrete query record so mashumaro can serialize EventQA questions."""


@dataclass(frozen=True)
class EventQAExperiment(Experiment[EventQAQuestion]):
    """Concrete experiment artifact for EventQA benchmark runs."""

    results: List[EventQAQueryResult]
