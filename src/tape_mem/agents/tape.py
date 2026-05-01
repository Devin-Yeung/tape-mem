from typing import Iterable, cast
import chromadb
from chromadb.api import ClientAPI

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
    __slots__ = ("_llm", "_template", "_tokenizor", "_active_tape", "_collection")

    def __init__(
        self,
        provider: ProviderConfig,
        template: Template,
        chroma_client: ClientAPI = chromadb.EphemeralClient(),
    ):
        self._setup_llm_backend(provider)
        self._template = template
        # for token usage metric estimation
        self._tokenizor = tiktoken.get_encoding("o200k_base")
        # create a new tape and write on it
        self._active_tape = self._llm.tape("main")
        # create a dedicated collection
        self._collection = chroma_client.create_collection(name="tape_mem_collection")

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

    def memorize_conversation(self, sessions: Iterable[ConversationSession]) -> None:
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
            logger.info(f"handoff session: {session_id}")
            self._active_tape.handoff(
                session_id,
                state={
                    "memorize": True,
                    "chat_time": session.chat_time,
                },
            )

            # Store each message with chat_time in metadata for cleaner separation
            for idx, msg in enumerate(session.messages):
                msg_id = f"{session_id}_msg_{idx}"

                logger.debug(f"adding message to tape: id={msg_id}, role={msg.role}")
                self._active_tape.append(
                    TapeEntry.message(
                        {"role": msg.role, "content": msg.content},
                        chat_time=session.chat_time,
                    )
                )

                logger.debug(
                    f"adding message to chroma db: id={msg_id}, role={msg.role}"
                )
                # store in the chroma db with metadata for session mapping
                # chat_time must be string for chromadb - datetime not accepted
                self._collection.add(
                    ids=msg_id,
                    documents=msg.content,
                    metadatas={
                        "session_id": session_id,
                        "chat_time": str(session.chat_time),
                        "role": msg.role,
                    },
                )

    def query(self, question: str, top_k: int = 10) -> AgentResponse:
        # 1. Query ChromaDB for relevant messages
        results = self._collection.query(
            query_texts=[question],
            n_results=top_k,
        )

        # 2. Extract unique session_ids from metadata
        session_ids: set[str] = set()
        if results["metadatas"] and results["metadatas"][0]:
            for meta in results["metadatas"][0]:
                session_ids.add(cast(str, meta["session_id"]))

        logger.debug(
            f"retrieved {len(results['metadatas'][0]) if results['metadatas'] and results['metadatas'][0] else 0} messages from {len(session_ids)} sessions"
        )

        # 3. For each session_id, fetch full session from tape to preserve locality
        context_messages: list[dict[str, str]] = []
        for sid in session_ids:
            entries = self._active_tape.query.after_anchor(sid).kinds("message").all()
            for entry in entries:
                context_messages.append(entry.payload)

        # 4. Estimate context token count
        context_str = "\n".join(m["content"] for m in context_messages)
        context_token_count = len(self._tokenizor.encode(context_str))

        # 5. Build messages with context prepended
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that answers questions using only the "
                    "memorized context when it is relevant."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context_str}\n\nQuestion: {question}",
            },
        ]

        # 6. Stream response
        stream = self._llm.stream(messages=messages)
        resp = "".join(list(stream))

        # 7. Return with stats
        usage = stream.usage
        if usage:
            logger.debug("token usage: {}", usage)
            stats = Stats(
                estimated_context_tokens=context_token_count,
                total_input_tokens=usage.get("input_tokens"),
                cache_read_tokens=(
                    usage.get("input_tokens_details", {}).get("cached_tokens", 0)
                ),
            )
            return AgentResponse(resp, QueryMetadata(stats=stats))
        else:
            return AgentResponse(resp)
