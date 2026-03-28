from dataclasses import dataclass
from typing import Optional, TypeVar, Generic

from mashumaro.mixins.json import DataClassJSONMixin
from mashumaro.config import BaseConfig

T = TypeVar("T")


@dataclass(frozen=True)
class Turn(Generic[T], DataClassJSONMixin):
    """
    Represents a single turn in a conversation, consisting of a user input and an agent response.
    """

    user: str
    agent: str
    metadata: Optional[T] = None

    class Config(BaseConfig):
        omit_none = True
