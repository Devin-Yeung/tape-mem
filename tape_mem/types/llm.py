from dataclasses import dataclass
from typing import Protocol, runtime_checkable, Literal, List


@dataclass(frozen=True)
class Message:
    """A single message in a chat conversation."""

    role: Literal["system", "user", "assistant"]
    content: str


@runtime_checkable
class LLM(Protocol):
    """Abstraction over LLM chat backends.

    Implementors must provide a chat() method that takes a list of
    messages and returns the model's response as a string.
    """

    def chat(self, messages: List[Message]) -> str:
        """Generate a chat completion from the given messages."""
        ...
