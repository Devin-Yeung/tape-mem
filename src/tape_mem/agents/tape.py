import tiktoken
import re
from collections.abc import Iterable, Sequence

from loguru import logger
from rank_bm25 import BM25Okapi
from republic import LLM, TapeEntry

from tape_mem.dataset.templates import Template
from tape_mem.types import Agent
from tape_mem.types.agent import (
    AgentResponse,
    ConversationSession,
    Stats,
    QueryMetadata,
)
from tape_mem.types.provider import ProviderConfig


# ==============================================================================
# Query Construction And Retrieval Heuristics
# ==============================================================================
#
# Republic's default tape context only replays messages after the last anchor.
# This agent writes a new anchor for every memorized chunk, so relying on
# `tape.chat(question)` makes retrieval far narrower than intended.
#
# Instead, we explicitly retrieve memorized chunks ourselves:
#
# 1. Reduce the question to a compact vocabulary list.
# 2. Let BM25 rank each memorized chunk against those terms.
# 3. Rebuild a prompt from the top-scoring chunks and answer the original question.
#
# This keeps the retrieval model easy to explain and debug.


_WORD_PATTERN = re.compile(r"[a-z0-9_/-]+")
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "answer",
    "are",
    "based",
    "be",
    "did",
    "do",
    "does",
    "for",
    "from",
    "give",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "please",
    "question",
    "tell",
    "that",
    "the",
    "their",
    "them",
    "these",
    "this",
    "those",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}
_MAX_QUERY_TERMS = 8
_MAX_RETRIEVED_CHUNKS = 8


def _word_tokens(text: str) -> list[str]:
    return _WORD_PATTERN.findall(text.lower())


def _dedupe_preserving_order(tokens: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        result.append(token)
    return result


def _format_context_chunks(chunks: list[dict]) -> str:
    """Format retrieved chunks with role and session information preserved.

    Each chunk is a dict with keys: role, content, chat_time.
    """
    formatted_chunks = []
    for idx, chunk in enumerate(chunks, start=1):
        role = chunk.get("role", "user")
        content = chunk.get("content", "")
        chat_time = chunk.get("chat_time", "")

        # Include session header if available
        if chat_time:
            header = f"[Session: {chat_time}]"
        else:
            header = ""

        # Format: include role marker for clarity
        formatted_content = f"{role}: {content}" if content else content
        chunk_text = f"{header}\n{formatted_content}" if header else formatted_content

        formatted_chunks.append(
            f'<chunk index="{idx}" role="{role}">\n{chunk_text}\n</chunk>'
        )

    return (
        "<retrieved_context>\n"
        + "\n\n".join(formatted_chunks)
        + "\n</retrieved_context>"
    )


class TapeAgent(Agent):
    def __init__(self, provider: ProviderConfig, template: Template):
        self._template = template

        model = f"openai:{provider.model}"
        self._llm = LLM(
            model=model,
            api_key={
                "openai": provider.api_key,
            },
            api_base={
                "openai": provider.base_url,
            },
        )
        logger.info(f"using model: {model}")
        logger.info(f"using base url: {provider.base_url}")

        # for token usage metric estimation
        self._tokenizor = tiktoken.get_encoding("o200k_base")

        # create a new tape and write on it
        self._active_tape = self._llm.tape("main")
        self._counter: int = 0

    def memorize(self, chunk: str) -> None:
        self._active_tape.handoff(f"chunk_{self._counter}", state={"memorize": True})
        self._counter += 1

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
        with their role and content preserved. The session's chat_time is embedded
        in each message's content for retrieval purposes.

        Args:
            sessions: Structured conversation sessions. Each session must have
                     chat_time (str) and messages (Sequence with role/content) attributes.
        """
        for session in sessions:
            session_id = f"session_{self._counter}"
            self._active_tape.handoff(
                session_id,
                state={
                    "type": "conversation_session",
                    "chat_time": session.chat_time,
                },
            )
            self._counter += 1

            # Store each message with chat_time embedded in content
            # This allows retrieval to include session context
            for msg in session.messages:
                content = f"[Session: {session.chat_time}]\n{msg.role}: {msg.content}"
                self._active_tape.append(
                    TapeEntry.message({"role": msg.role, "content": content})
                )
            # Add assistant acknowledgment
            self._active_tape.append(
                TapeEntry.message(
                    {
                        "role": "assistant",
                        "content": "I have learned the conversation and I will answer the question you ask.",
                    }
                )
            )

    # ==============================================================================
    # Retrieval Query Construction
    # ==============================================================================
    #
    # We separate answer generation from retrieval. Retrieval works on a short
    # vocabulary list so generic phrasing in the question does not dilute scoring.
    def _build_query_terms(self, question: str) -> list[str]:
        raw_tokens = _word_tokens(question)
        if not raw_tokens:
            return []

        significant_tokens = [
            token for token in raw_tokens if token not in _STOP_WORDS and len(token) > 1
        ]
        selected_tokens = significant_tokens or raw_tokens
        deduped_tokens = _dedupe_preserving_order(selected_tokens)
        return deduped_tokens[:_MAX_QUERY_TERMS]

    def _memorized_chunks(self) -> list[dict]:
        """Returns list of {role, content, chat_time} dicts for retrieved chunks."""
        messages = self._active_tape.query.kinds("message").all()
        chunks: list[dict] = []
        for entry in messages:
            payload = entry.payload
            if not isinstance(payload, dict):
                continue
            if payload.get("role") != "user":
                continue
            content = payload.get("content")
            if not isinstance(content, str) or not content.strip():
                continue

            # Parse chat_time from content prefix: "[Session: {chat_time}]\n..."
            chat_time = ""
            if content.startswith("[Session:"):
                end_idx = content.find("]\n")
                if end_idx != -1:
                    chat_time = content[
                        9:end_idx
                    ]  # Extract text between "[Session: " and "]\n"
                    # Also extract the actual message content after the prefix
                    content = content[end_idx + 2 :]

            # Also parse role from content if present: "{role}: {actual_content}"
            role = payload.get("role", "user")
            if content and ": " in content:
                role_part, _, actual_content = content.partition(": ")
                # Only override role if it looks like a valid role name
                if role_part in ("user", "assistant", "system"):
                    role = role_part
                    content = actual_content

            chunks.append(
                {
                    "role": role,
                    "content": content,
                    "chat_time": chat_time,
                }
            )
        return chunks

    def _retrieve_chunks(self, question: str) -> list[dict]:
        """Retrieve chunks preserving structured data (role, content, chat_time)."""
        query_terms = self._build_query_terms(question)
        if not query_terms:
            return []

        chunks = self._memorized_chunks()
        if not chunks:
            return []

        # For BM25, we tokenize the content (which now includes chat_time prefix)
        unique_chunks: list[dict] = []
        tokenized_chunks: list[list[str]] = []
        seen_chunks: set[str] = set()
        for chunk in chunks:
            # Use role + content for deduplication
            dedupe_key = f"{chunk.get('role', '')}:{chunk.get('content', '')}"
            normalized_chunk = chunk.get("content", "").strip()
            if not normalized_chunk or dedupe_key in seen_chunks:
                continue
            seen_chunks.add(dedupe_key)
            tokens = _word_tokens(normalized_chunk)
            if not tokens:
                continue
            unique_chunks.append(chunk)
            tokenized_chunks.append(tokens)

        if not tokenized_chunks:
            return []

        bm25 = BM25Okapi(tokenized_chunks)
        scores = bm25.get_scores(query_terms)

        ranked_indices = sorted(
            range(len(unique_chunks)),
            key=lambda idx: float(scores[idx]),
            reverse=True,
        )

        retrieved_chunks: list[dict] = []
        for idx in ranked_indices:
            if float(scores[idx]) <= 0:
                continue
            retrieved_chunks.append(unique_chunks[idx])
            if len(retrieved_chunks) >= _MAX_RETRIEVED_CHUNKS:
                break
        return retrieved_chunks

    def query(self, question: str) -> AgentResponse:
        retrieved_chunks = self._retrieve_chunks(question)

        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that answers questions using only the "
                    "memorized context when it is relevant."
                ),
            }
        ]

        if retrieved_chunks:
            tagged_context = _format_context_chunks(retrieved_chunks)
            context_token_count = len(self._tokenizor.encode(tagged_context))
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "The following memorized context was retrieved because it may be "
                        "relevant to the question. Each chunk is wrapped in XML tags so "
                        f"its boundaries stay explicit.\n\n{tagged_context}"
                    ),
                }
            )
        else:
            # TODO: Fall back to a broader scan or semantic retrieval when no chunk
            # clears these lexical heuristics. For now we leave the omission
            # explicit so low-recall cases are easier to reason about.
            context_token_count = 0
            logger.debug("no retrieved chunks matched question: {}", question)

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
