from collections.abc import Sequence

import tiktoken
from loguru import logger
from republic import LLM, TapeEntry

from tape_mem.dataset.templates import Template
from tape_mem.types import Agent
from tape_mem.types.agent import (
    AgentResponse,
    ConversationSession,
    QueryMetadata,
    Stats,
)
from tape_mem.types.provider import ProviderConfig
import uuid


class TapeAgent(Agent):
    __slots__ = ("_llm", "_template", "_tokenizor", "_active_tape")

    def __init__(self, provider: ProviderConfig, template: Template):
        self._setup_llm_backend(provider)
        self._template = template
        # for token usage metric estimation
        self._tokenizor = tiktoken.get_encoding("o200k_base")
        # create a new tape and write on it
        self._active_tape = self._llm.tape("main")

    def _setup_llm_backend(self, provider: ProviderConfig):
        model = f"openai:{provider.model}"
        self._llm = LLM(
            model=model,
            api_key={"openai": provider.api_key},
            api_base={"openai": provider.base_url},
        )
        logger.info(f"using model: {model}")
        logger.info(f"using base url: {provider.base_url}")

    def memorize(self, chunk: str) -> None:
        self._active_tape.handoff(f"chunk_{uuid.uuid6()}", state={"memorize": True})

        # write the chunk on tape
        self._active_tape.append(
            TapeEntry.message(
                # todo: using template (blocking by template refactoring)
                {"role": "user", "content": chunk}
            )
        )
        # pretend that agent responses
        self._active_tape.append(
            TapeEntry.message(
                # todo: using template (blocking by template refactoring)
                {
                    "role": "assistant",
                    "content": "I have learned the documents and I will answer the question you ask.",
                }
            )
        )

    def forget(self, chunk: str) -> None:
        raise NotImplementedError

    def memorize_conversation(self, sessions: Sequence[ConversationSession]) -> None:
        """Memorize structured conversation sessions using handoff for session boundaries.

        Each session gets its own handoff anchor. Messages are stored individually
        with their role and content preserved. The session's chat_time is stored
        in each message's metadata for clean separation.

        Args:
            sessions: Structured conversation sessions. Each session must have
                     chat_time (str) and messages (Sequence with role/content) attributes.
        """
        for session in sessions:
            session_id = f"session_{uuid.uuid6()}"
            self._active_tape.handoff(
                session_id,
                state={
                    "memorize": True,
                    "chat_time": session.chat_time,
                },
            )

            # Store each message with chat_time in metadata for cleaner separation
            for msg in session.messages:
                self._active_tape.append(
                    TapeEntry.message(
                        {"role": msg.role, "content": msg.content},
                        chat_time=session.chat_time,
                    )
                )

    def query(self, question: str) -> AgentResponse:
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that answers questions using only the "
                    "memorized context when it is relevant."
                ),
            }
        ]

        # TODO: retrieve context
        context_token_count: int = 0

        messages.append(
            {
                "role": "user",
                "content": (
                    "Answer the question based on the memorized documents. Give me the "
                    f"answer directly without any explanation.\nQuestion: {question}"
                ),
            }
        )

        stream = self._llm.stream(messages=messages)
        resp = "".join(list(stream))

        usage = stream.usage
        if usage:
            logger.debug("token usage: {}", usage)
            stats = Stats(
                estimated_context_tokens=context_token_count,
                total_input_tokens=(usage.get("input_tokens")),
                cache_read_tokens=(
                    usage.get("input_tokens_details", {}).get("cached_tokens", 0)
                ),
            )
            meta = QueryMetadata(stats=stats)
            return AgentResponse(resp, meta)
        else:
            return AgentResponse(resp)
