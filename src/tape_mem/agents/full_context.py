from pydantic import BaseModel

from tape_mem.dataset.templates import SYSTEM_MESSAGE, Template
from typing import List

from tape_mem.types import Agent
from mirascope.llm import Message, Response
from mirascope import llm

from tape_mem.types.agent import AgentResponse


class QueryResponse(BaseModel):
    answer: str


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

    def query(self, question: str) -> AgentResponse:
        msg: List[llm.Message] = []

        msg.extend(self._mem)
        msg.extend(self._template.query_template(question))

        resp: Response[QueryResponse] = self._model.call(
            msg, format=llm.format(QueryResponse, mode="json")
        )  # ty:ignore[invalid-assignment]
        result, resp = resp.validate(max_retries=3)

        return AgentResponse(answer=result.answer)
