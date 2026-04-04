from typing import Literal

from tape_mem.dataset.templates import get_template
from tape_mem.types import Agent, LLM, Message
from tape_mem.types.agent import AgentVariant


class FullContextAgent(Agent):
    """A v0 baseline agent that replays every memorized chunker on each query."""

    def __init__(
        self,
        *,
        dataset_variant: Literal[
            "ruler_qa",
            "longmemeval",
            "eventqa",
            "in_context_learning",
            "recsys_redial",
            "infbench_sum",
            "detective_qa",
            "factconsolidation",
        ],
        llm: LLM,
        agent_variant: AgentVariant = "long_context_agent",
    ) -> None:
        self._dataset_variant = dataset_variant
        self._llm = llm
        self._agent_variant = agent_variant
        self._memory_chunks: list[str] = []

    def memorize(self, chunk: str) -> None:
        self._memory_chunks.append(chunk)

    def forget(self, chunk: str) -> None:
        try:
            self._memory_chunks.remove(chunk)
        except ValueError:
            return

    def query(self, question: str) -> str:
        messages = self._build_messages(question)
        response = self._llm.chat(messages)

        if response == "":
            raise ValueError("Received empty response from LLM backend")

        return response

    def _build_messages(self, question: str) -> list[Message]:
        # We replay the full memorize history on every query to keep the v0
        # baseline faithful to the benchmark's inject-once/query-many semantics.
        messages = [
            Message(
                role="system",
                content=get_template(
                    self._dataset_variant,
                    "system",
                    self._agent_variant,
                ),
            )
        ]

        # TODO: Replace full replay with explicit memory retrieval or compression
        # once the repository grows beyond the long-context baseline.
        memorize_template = get_template(
            self._dataset_variant,
            "memorize",
            self._agent_variant,
        )
        for chunk in self._memory_chunks:
            messages.append(
                Message(
                    role="user",
                    content=memorize_template.format(
                        context=chunk,
                        time_stamp="",
                    ),
                )
            )

        messages.append(
            Message(
                role="user",
                content=get_template(
                    self._dataset_variant,
                    "query",
                    self._agent_variant,
                ).format(question=question),
            )
        )

        return messages
