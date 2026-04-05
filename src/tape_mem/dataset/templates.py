# source: https://github.com/HUST-AI-HYZ/MemoryAgentBench/blob/main/utils/templates.py
from mirascope import llm
from typing import List, Protocol, runtime_checkable

# System message used across all templates
SYSTEM_MESSAGE = "You are a helpful assistant that can read the context and memorize it for future retrieval."


@runtime_checkable
class Template(Protocol):
    def memorize_template(self, chunk: str) -> List[llm.Message]: ...

    def query_template(self, question: str) -> List[llm.Message]: ...


class EventQATemplate:
    def memorize_template(self, chunk: str) -> List[llm.Message]:
        """Generate the memorize template for EventQA dataset with a given timestamp."""
        return [
            llm.messages.user(
                f"The following context is the documents I have read:\n{chunk}"
            ),
            llm.messages.assistant(
                "I have learned the documents and I will answer the question you ask.",
                model_id=None,
                provider_id=None,
            ),
        ]

    def query_template(self, question: str) -> List[llm.Message]:
        return [
            llm.messages.user(
                f"Answer the question based on the memorized documents. Give me the answer directly without any explanation.\nQuestion: {question}"
            )
        ]
