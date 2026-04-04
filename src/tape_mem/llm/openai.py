from attr import dataclass
from typing import List, assert_never

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
)

from tape_mem.types.llm import LLM, Message


@dataclass
class OpenAIConfig:
    """OpenAI configuration for LLM.

    Attributes:
        endpoint: Base URL of the OpenAI-compatible API (e.g. "https://api.openai.com/v1").
        api_key: API key for authentication.
        model: Model identifier (e.g. "gpt-4o-mini", "gpt-5.2").
    """

    endpoint: str
    api_key: str
    model: str


class OpenAIChat(LLM):
    """OpenAI-compatible chat implementation using the official SDK."""

    def __init__(self, config: OpenAIConfig):

        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.endpoint,
        )
        self._model = config.model

    def chat(self, messages: List[Message]) -> str:
        msgs: List[ChatCompletionMessageParam] = []

        for message in messages:
            match message.role:
                case "system":
                    msg: ChatCompletionSystemMessageParam = {
                        "content": message.content,
                        "role": message.role,
                    }
                    msgs.append(msg)
                case "user":
                    msg: ChatCompletionUserMessageParam = {
                        "content": message.content,
                        "role": message.role,
                    }
                    msgs.append(msg)
                case "assistant":
                    msg: ChatCompletionAssistantMessageParam = {
                        "content": message.content,
                        "role": message.role,
                    }
                    msgs.append(msg)
                case _:
                    assert_never(message.role)

        response = self._client.chat.completions.create(
            model=self._model,
            messages=msgs,
        )

        content = response.choices[0].message.content

        if content is None:
            raise ValueError("Received empty response from OpenAI API")

        return content
