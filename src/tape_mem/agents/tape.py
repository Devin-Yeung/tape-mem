from tape_mem.types.conversation import Session, Message
import hashlib
import uuid
from typing import Optional, cast, Collection

import chromadb
import tiktoken
from chromadb.api import ClientAPI
from loguru import logger
from republic import LLM, TapeEntry

from tape_mem.dataset.templates import Template
from tape_mem.types import Agent
from tape_mem.types.agent import (
    AgentResponse,
    AbstractSession,
    QueryMetadata,
    Stats,
)
from tape_mem.types.provider import ProviderConfig


class TapeAgent(Agent):
    __slots__ = (
        "_llm",
        "_template",
        "_tokenizor",
        "_active_tape",
        "_collection",
        "_anchors",
    )

    def __init__(
        self,
        provider: ProviderConfig,
        template: Template,
        chroma_client: ClientAPI | None = None,
    ):
        self._setup_llm_backend(provider)
        self._template = template
        # for token usage metric estimation
        self._tokenizor = tiktoken.get_encoding("o200k_base")
        # create a new tape and write on it
        self._active_tape = self._llm.tape("main")
        # create a dedicated collection with a unique name to avoid collisions
        # when multiple TapeAgent instances share an EphemeralClient
        # todo: collection should be tied to the database, so we can reused the collection across different runs with the same database (e.g. persisted one) and still avoid collision. Current implementation will create a new collection for each agent instance, which is not ideal.
        if chroma_client is None:
            chroma_client = chromadb.EphemeralClient()
        collection_name = f"tape_mem_collection_{uuid.uuid4().hex[:8]}"
        self._collection = chroma_client.get_or_create_collection(name=collection_name)
        # list of anchor (in the order of tape store)
        self._anchors: list[str] = []

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
        # todo: reuse memorize conversation
        session = Session(
            messages=(
                Message(role="user", content=chunk),
                Message(
                    role="assistant",
                    content="I have learned the documents and I will answer the question you ask.",
                ),
            )
        )
        self.memorize_conversation([session])

    def forget(self, chunk: str) -> None:
        raise NotImplementedError

    def _get_next_anchor(self, sid: str) -> Optional[str]:
        """Get the next anchor (sid) for a conversation session.

        Returns next anchor name or None if sid not found or sid is the last anchor.
        """
        try:
            idx = self._anchors.index(sid)
        except ValueError:
            # sid not in list
            return None

        # return next if exists
        if idx + 1 < len(self._anchors):
            return self._anchors[idx + 1]
        return None

    def _message_id(self, session_id: str, msg: str) -> str:
        """Generate a unique message ID based on session_id and message content."""
        # hash content and then hash session_id
        hasher = hashlib.sha256()
        hasher.update(session_id.encode("utf-8"))
        hasher.update(msg.encode("utf-8"))
        return hasher.hexdigest()

    def memorize_conversation(self, sessions: Collection[AbstractSession]) -> None:
        """Memorize structured conversation sessions using handoff for session boundaries.

        Each session gets its own handoff anchor. Messages are stored individually
        with their role and content preserved. The session's chat_time is stored
        in each message's metadata for clean separation.

        Args:
            sessions: Structured conversation sessions. Each session must have
                     chat_time (str) and messages (Sequence with role/content) attributes.
        """
        for session in sessions:
            session_id = session.session_id
            self._anchors.append(session.session_id)
            logger.info(f"handoff session: {session_id}")

            self._active_tape.handoff(
                session_id,
                state={
                    "memorize": True,
                    "chat_time": session.chat_time,
                },
            )

            # Store each message with chat_time in metadata for cleaner separation
            logger.info(
                f"adding {len(session.messages)} to tape and chroma db for session_id={session_id}"
            )
            for idx, msg in enumerate(session.messages):
                msg_id = self._message_id(session_id, msg.content)

                self._active_tape.append(
                    TapeEntry.message(
                        {"role": msg.role, "content": msg.content},
                        chat_time=session.chat_time,
                    )
                )

                # store in the chroma db with metadata for session mapping
                # chat_time must be string for chromadb - datetime not accepted
                self._collection.add(
                    ids=msg_id,
                    documents=msg.content,
                    metadatas={
                        "session_id": session_id,
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
            next_sid = self._get_next_anchor(sid)
            logger.info(
                f"fetching messages between session_id={sid} next_sid={next_sid}"
            )

            if next_sid is None:
                entries = (
                    self._active_tape.query.after_anchor(sid).kinds("message").all()
                )
            else:
                entries = (
                    self._active_tape.query.between_anchors(sid, next_sid)
                    .kinds("message")
                    .all()
                )
            for entry in entries:
                context_messages.append(entry.payload)

        # 4. Estimate context token count
        context_str = "\n".join(m["content"] for m in context_messages)
        context_token_count = len(self._tokenizor.encode(context_str))
        logger.debug(f"estimated context token count: {context_token_count}")

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
