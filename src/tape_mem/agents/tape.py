from tape_mem.dataset.templates import Template
from tape_mem.types import Agent
from tape_mem.types.agent import AgentResponse
from tape_mem.types.provider import ProviderConfig
from republic import LLM, TapeEntry


class TapeAgent(Agent):
    def __init__(self, provider: ProviderConfig, template: Template):
        self._template = template
        self._llm = LLM(
            model=f"openai:{provider.model}",
            api_key=provider.api_key,
            api_base=provider.base_url,
        )

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
                    "role": "user",
                    "content": "I have learned the documents and I will answer the question you ask.",
                }
            )
        )

    def forget(self, chunk: str) -> None:
        raise NotImplementedError

    def query(self, question: str) -> AgentResponse:
        raise NotImplementedError
