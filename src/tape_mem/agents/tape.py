from tape_mem.types import Agent
from tape_mem.types.agent import AgentResponse
from tape_mem.types.provider import ProviderConfig
from republic import LLM


class TapeAgent(Agent):
    def __init__(self, provider: ProviderConfig):
        self.llm = LLM(
            model=f"openai:{provider.model}",
            api_key=provider.api_key,
            api_base=provider.base_url,
        )

        # create a new tape and write on it
        self.llm.tape("main")

    def memorize(self, chunk: str) -> None:
        raise NotImplementedError

    def forget(self, chunk: str) -> None:
        raise NotImplementedError

    def query(self, question: str) -> AgentResponse:
        raise NotImplementedError
