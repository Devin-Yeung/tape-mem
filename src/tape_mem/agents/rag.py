from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, List

import tiktoken
from mirascope import llm
from mirascope.llm import Response
from pydantic import BaseModel

from tape_mem.dataset.templates import SYSTEM_MESSAGE, Template
from tape_mem.types import Agent
from tape_mem.types.agent import AgentResponse, QueryMetadata, Stats


class QueryResponse(BaseModel):
    answer: str


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


@dataclass
class _Doc:
    text: str
    tf: dict[str, int]
    length: int


class RagAgent(Agent):
    """
    A lightweight RAG agent with in-memory BM25-style retrieval.

    Notes:
    - This agent only adds a new file (no changes to existing types/CLI).
    - Metrics are limited to existing fields in `Stats` / `QueryMetadata`.
    """

    def __init__(
        self,
        model: llm.Model,
        template: Template,
        *,
        top_k: int = 8,
        max_context_tokens: int | None = 24_000,
        token_buffer: int = 512,
        bm25_k1: float = 1.2,
        bm25_b: float = 0.75,
        tokenizer_model_name: str = "gpt-4o-mini",
    ) -> None:
        self._model = model
        self._template = template
        self._top_k = top_k
        self._max_context_tokens = max_context_tokens
        self._token_buffer = token_buffer
        self._k1 = bm25_k1
        self._b = bm25_b

        self._docs: list[_Doc] = []
        self._df: dict[str, int] = {}
        self._total_len = 0

        self._encoding = self._init_encoding(tokenizer_model_name)

    def _init_encoding(self, model_name: str):
        try:
            return tiktoken.encoding_for_model(model_name)
        except KeyError:
            return tiktoken.encoding_for_model("gpt-4o-mini")

    def memorize(self, chunk: str) -> None:
        tokens = _tokenize(chunk)
        tf: dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1

        unique_terms = tf.keys()
        for term in unique_terms:
            self._df[term] = self._df.get(term, 0) + 1

        doc = _Doc(text=chunk, tf=tf, length=len(tokens))
        self._docs.append(doc)
        self._total_len += doc.length

    def forget(self, chunk: str) -> None:
        # Precise forgetting would require rebuilding DF; keep behavior explicit.
        raise NotImplementedError()

    def _avgdl(self) -> float:
        if not self._docs:
            return 0.0
        return self._total_len / len(self._docs)

    def _idf(self, term: str) -> float:
        # BM25 IDF with +1 inside log to avoid negative values on frequent terms.
        n = len(self._docs)
        df = self._df.get(term, 0)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def _score(self, doc: _Doc, q_terms: Iterable[str]) -> float:
        avgdl = self._avgdl()
        if avgdl <= 0.0:
            return 0.0

        score = 0.0
        for term in q_terms:
            f = doc.tf.get(term, 0)
            if f <= 0:
                continue
            idf = self._idf(term)
            denom = f + self._k1 * (1.0 - self._b + self._b * (doc.length / avgdl))
            score += idf * (f * (self._k1 + 1.0)) / denom
        return score

    def _retrieve(self, question: str) -> list[str]:
        if not self._docs:
            return []

        q_terms = _tokenize(question)
        if not q_terms:
            return []

        scored: list[tuple[float, int]] = []
        for i, doc in enumerate(self._docs):
            s = self._score(doc, q_terms)
            if s > 0:
                scored.append((s, i))

        scored.sort(reverse=True)
        top = scored[: self._top_k]
        return [self._docs[i].text for _, i in top]

    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self._encoding.encode(text, allowed_special={"<|endoftext|>"}))

    def _pack_retrieved(self, retrieved: list[str], question: str) -> list[str]:
        """
        Pack retrieved chunks into a token budget.

        Budget is approximate because chat message overhead is ignored,
        but this is sufficient to prevent grossly oversized prompts.
        """
        if self._max_context_tokens is None:
            return retrieved

        # Reserve space for query template and a buffer.
        query_msgs = self._template.query_template(question)
        query_tokens = 0
        for m in query_msgs:
            content = getattr(m, "content", None)
            if isinstance(content, str):
                query_tokens += self._count_tokens(content)

        remaining = self._max_context_tokens - self._token_buffer - query_tokens
        if remaining <= 0:
            return []

        packed: list[str] = []
        used = 0
        for chunk in retrieved:
            t = self._count_tokens(chunk)
            if t <= 0:
                continue
            if used + t > remaining:
                break
            packed.append(chunk)
            used += t
        return packed

    def _estimate_context_tokens(self, messages: list[llm.Message]) -> int:
        # Approximate: sum of tokenized content only (not counting role/system overhead).
        total = 0
        for m in messages:
            content = getattr(m, "content", None)
            if isinstance(content, str) and content:
                total += len(
                    self._encoding.encode(content, allowed_special={"<|endoftext|>"})
                )
        return total

    def _model_name(self) -> str | None:
        for attr in ("model", "model_id", "name"):
            v = getattr(self._model, attr, None)
            if isinstance(v, str) and v:
                return v
        s = str(self._model)
        return s if s else None

    def query(self, question: str) -> AgentResponse:
        retrieved = self._pack_retrieved(self._retrieve(question), question)

        msg: list[llm.Message] = [llm.messages.system(SYSTEM_MESSAGE)]
        for chunk in retrieved:
            msg.extend(self._template.memorize_template(chunk))
        msg.extend(self._template.query_template(question))

        estimated_context_tokens = self._estimate_context_tokens(msg)

        resp: Response[QueryResponse] = self._model.call(
            msg, format=llm.format(QueryResponse, mode="json")
        )  # ty:ignore[invalid-assignment]
        result, resp = resp.validate(max_retries=3)

        usage = resp.usage
        stats = Stats(
            estimated_context_tokens=estimated_context_tokens,
            total_input_tokens=(usage.input_tokens if usage is not None else None),
            cache_read_tokens=(usage.cache_read_tokens if usage is not None else None),
        )

        metadata = QueryMetadata(stats=stats, model_name=self._model_name())
        return AgentResponse(answer=result.answer, metadata=metadata)
