from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol, runtime_checkable, Iterable

from mashumaro.config import BaseConfig
from mashumaro.mixins.json import DataClassJSONMixin


# ==============================================================================
# Conversation Data Protocols
# ==============================================================================
#
# These protocols define the structure of conversation data for agents that
# support conversation-based datasets like LongMemEval.


class ConversationMessage(Protocol):
    """A single message in a conversation.

    The `role` is one of the common chat roles.
    """

    @property
    def role(self) -> Literal["user", "assistant", "system"]: ...

    @property
    def content(self) -> str: ...


class ConversationSession(Protocol):
    """One conversation session identified by a timestamp.

    `chat_time` is represented as a `datetime` for stronger typing and easier
    manipulation. Implementations that still use strings should accept them and
    convert to `datetime` as needed.
    """

    @property
    def chat_time(self) -> datetime: ...

    @property
    def messages(self) -> Iterable[ConversationMessage]: ...


@dataclass(frozen=True)
class Stats(DataClassJSONMixin):
    estimated_context_tokens: int | None = None
    total_input_tokens: int | None = None
    cache_read_tokens: int | None = None

    class Config(BaseConfig):
        omit_none = True


@dataclass(frozen=True)
class QueryMetadata(DataClassJSONMixin):
    stats: Stats | None = None
    model_name: str | None = None

    class Config(BaseConfig):
        omit_none = True


@dataclass(frozen=True)
class AgentResponse(DataClassJSONMixin):
    answer: str
    metadata: QueryMetadata | None = None

    class Config(BaseConfig):
        omit_none = True


@runtime_checkable
class Agent(Protocol):
    """A protocol defining the interface for a memory-augmented agent.

    The agent maintains an internal memory store that can be populated via
    memorize(), selectively cleared via forget(), and queried via query().

    Implementations are expected to be stateful — calls to memorize() should
    accumulate within the agent's memory across invocations, until explicitly
    forgotten or the agent is re-initialized.
    """

    def memorize(self, chunk: str) -> None:
        """Add a piece of information to the agent's memory store.

        The agent should append or index the chunk so it can be retrieved
        later via query(). Implementations may choose to deduplicate or
        merge semantically similar chunks.
        """
        ...

    def forget(self, chunk: str) -> None:
        """Remove a previously memorized chunk from the agent's memory.

        If the exact chunk is not found, implementations should silently
        succeed (no-op) rather than raise an error. Not all implementations
        may support precise forgetting — in that case, this method may raise
        NotImplementedError.
        """
        ...

    def memorize_conversation(self, sessions: Sequence[ConversationSession]) -> None:
        """Memorize structured conversation sessions.

        Default implementation serializes sessions to plain text and calls
        memorize() for each session. Agents with native conversation support
        (e.g., TapeAgent with handoff) should override this method to preserve
        the structured conversation data (role, content, timestamps).

        Args:
            sessions: Structured conversation sessions. Each session must have
                     chat_time (datetime) and messages (Sequence[ConversationMessage])
                     attributes.
        """
        for session in sessions:
            chunk = self._serialize_session(session)
            self.memorize(chunk)

    def _serialize_session(self, session: ConversationSession) -> str:
        """Serialize a session to text format.

        Override this method to customize serialization format.

        Args:
            session: A session object with chat_time and messages attributes.

        Returns:
            Serialized text representation of the session.
        """
        # Ensure the timestamp is formatted as text when serializing.
        time_str = session.chat_time.isoformat()

        lines = [time_str]
        for msg in session.messages:
            lines.append(f"{msg.role}: {msg.content}")
        return "\n".join(lines)

    def query(self, question: str) -> AgentResponse:
        """Ask the agent a question using its accumulated memory as context.

        The agent should answer based on information previously stored via
        memorize(). The question itself should NOT be added to memory —
        only the answer (if any) should be stored if the implementation
        chooses to do so.

        Args:
            question: The question to ask the agent.

        Returns:
            The agent's answer as a string. If no relevant information is
            found in memory, implementations may return an empty string or
            a message indicating the information is unknown.
        """
        ...
