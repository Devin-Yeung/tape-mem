import math
import re
from collections import Counter
from collections.abc import Iterable

from loguru import logger
from republic import LLM, TapeEntry

from tape_mem.dataset.templates import Template
from tape_mem.types import Agent
from tape_mem.types.agent import AgentResponse
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
# 2. Score each memorized chunk by how well it covers those terms.
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

    def _memorized_chunks(self) -> list[str]:
        messages = self._active_tape.query.kinds("message").all()
        chunks: list[str] = []
        for entry in messages:
            payload = entry.payload
            if not isinstance(payload, dict):
                continue
            if payload.get("role") != "user":
                continue
            content = payload.get("content")
            if isinstance(content, str) and content.strip():
                chunks.append(content)
        return chunks

    def _document_frequency(self, chunks: list[str]) -> Counter[str]:
        counts: Counter[str] = Counter()
        for chunk in chunks:
            counts.update(set(_word_tokens(chunk)))
        return counts

    def _score_chunk(
        self,
        question: str,
        query_terms: list[str],
        document_frequency: Counter[str],
        document_count: int,
        chunk: str,
    ) -> float:
        normalized_chunk = chunk.lower()
        normalized_question = question.strip().lower()
        chunk_terms = set(_word_tokens(normalized_chunk))

        score = 0.0
        if normalized_question and normalized_question in normalized_chunk:
            score += 4.0
        if query_terms:
            exact_phrase = " ".join(query_terms)
            if exact_phrase and exact_phrase in normalized_chunk:
                score += 2.0

        matched_terms = [term for term in query_terms if term in chunk_terms]
        if query_terms:
            score += len(matched_terms)
            score += len(matched_terms) / len(query_terms)

        # Rare terms carry more retrieval signal than common ones, so we weight
        # them slightly higher instead of treating every token equally.
        for term in matched_terms:
            score += math.log(1.0 + document_count / document_frequency[term])

        return score

    def _retrieve_chunks(self, question: str) -> list[str]:
        query_terms = self._build_query_terms(question)
        if not query_terms:
            return []

        chunks = self._memorized_chunks()
        if not chunks:
            return []

        document_frequency = self._document_frequency(chunks)
        document_count = len(chunks)
        scored_chunks: list[tuple[float, str]] = []
        seen_chunks: set[str] = set()

        for chunk in chunks:
            normalized_chunk = chunk.strip()
            if not normalized_chunk or normalized_chunk in seen_chunks:
                continue
            seen_chunks.add(normalized_chunk)

            score = self._score_chunk(
                question=question,
                query_terms=query_terms,
                document_frequency=document_frequency,
                document_count=document_count,
                chunk=normalized_chunk,
            )
            if score <= 0:
                continue
            scored_chunks.append((score, normalized_chunk))

        scored_chunks.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in scored_chunks[:_MAX_RETRIEVED_CHUNKS]]

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
            joined_context = "\n\n".join(
                f"Context {idx}:\n{chunk}"
                for idx, chunk in enumerate(retrieved_chunks, start=1)
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "The following memorized context was retrieved because it may be "
                        f"relevant to the question.\n\n{joined_context}"
                    ),
                }
            )
        else:
            # TODO: Fall back to a broader scan or semantic retrieval when no chunk
            # clears these lexical heuristics. For now we leave the omission
            # explicit so low-recall cases are easier to reason about.
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

        resp = self._llm.chat(messages=messages)
        return AgentResponse(resp)
