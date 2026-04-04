from tape_mem.dataset.templates import SYSTEM_MESSAGE, Template
from typing import List

from tape_mem.types import Agent
from mirascope.llm import Message
from mirascope import llm


class FullContextAgent(Agent):
    """A v0 baseline agent that replays every memorized chunk on each query."""

    def __init__(
        self,
        model: llm.Model,
        template: Template,
    ) -> None:
        self._model = model
        self._template = template
        self._mem: List[Message] = [llm.messages.system(SYSTEM_MESSAGE)]

    def memorize(self, chunk: str) -> None:
        msg = self._template.memorize_template(chunk)
        self._mem.extend(msg)

    def forget(self, chunk: str) -> None:
        # TODO: implement forget
        raise NotImplementedError()

    def query(self, question: str) -> str:
        msg: List[llm.Message] = []

        msg.extend(self._mem)
        msg.extend(self._template.query_template(question))

        resp = self._model.call(msg)

        # todo: only return the first text msg?
        for text in resp.texts:
            return text.text

        return ""
