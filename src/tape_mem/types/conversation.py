import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Tuple

from mashumaro.config import BaseConfig
from mashumaro.mixins.json import DataClassJSONMixin


@dataclass(frozen=True)
class Message(DataClassJSONMixin):
    """A minimal concrete ConversationMessage implementation.

    Attributes:
        role: One of "user", "assistant", or "system".
        content: The message text.
    """

    role: Literal["user", "assistant", "system"]
    content: str

    class Config(BaseConfig):
        omit_none = True


@dataclass(frozen=True)
class Session(DataClassJSONMixin):
    """A simple concrete ConversationSession implementation.

    This implementation generates a deterministic, SHA-256 based
    `session_id` from the session timestamp and the sequence of messages.
    The `messages` passed in may be any sequence; they are normalized to a
    tuple on construction so the dataclass is fully immutable.
    """

    messages: Tuple[Message, ...] = field(default_factory=tuple)
    chat_time: datetime | None = None
    session_id: str = field(init=False)

    class Config(BaseConfig):
        omit_none = True

    def __post_init__(self) -> None:
        # Normalize messages to a tuple of ConversationMessage instances so the
        # dataclass becomes fully immutable. If the provided items are already
        # ConversationMessage instances we keep them; otherwise we attempt to
        # construct ConversationMessage from objects exposing `role` and
        # `content` attributes.
        msgs: Tuple[Message, ...] = tuple(
            m if isinstance(m, Message) else Message(role=m.role, content=m.content)
            for m in self.messages
        )
        object.__setattr__(self, "messages", msgs)

        # Compute a deterministic sha256 hex digest as the session id.
        hasher = hashlib.sha256()

        for m in msgs:
            hasher.update(m.role.encode("utf-8"))
            hasher.update(m.content.encode("utf-8"))

        object.__setattr__(self, "session_id", hasher.hexdigest())
