from tape_mem.dataset.templates import (
    SYSTEM_MESSAGE,
    memorize_eventqa_chunk,
    query_eventqa,
)
from typing import List

from tape_mem.types import Agent
from mirascope.llm import Message
from mirascope import llm


class FullContextAgent(Agent):
    """A v0 baseline agent that replays every memorized chunk on each query."""

    def __init__(
        self,
        dataset_name: str,
        model: llm.Model,
    ) -> None:
        self._dataset_name = dataset_name
        self._model = model
        self._mem: List[Message] = [llm.messages.system(SYSTEM_MESSAGE)]

    def memorize(self, chunk: str) -> None:
        # TODO: consider to use different template for different dataset
        msg = memorize_eventqa_chunk(chunk)
        self._mem.extend(msg)

    def forget(self, chunk: str) -> None:
        # TODO: implement forget
        raise NotImplementedError()

    def query(self, question: str) -> str:
        msg: List[llm.Message] = []

        msg.extend(self._mem)
        msg.extend(query_eventqa(question))

        resp = self._model.call(msg)

        # todo: only return the first text msg?
        for text in resp.texts:
            return text.text

        return ""
