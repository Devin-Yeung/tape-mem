from tape_mem.dataset.templates import EventQATemplate
from tape_mem.chunker import SentenceAwareChunker
from tape_mem.agents import FullContextAgent
from tape_mem.dataset.eventqa import naive_eventqa_example
from typing import Sequence
from mirascope import llm

from .settings.env import Env
from loguru import logger


# The CLI package owns the command surface so the library package can stay
# import-focused under the new two-package src layout.
def main(argv: Sequence[str] | None = None) -> int:
    env = Env()  # ty:ignore[missing-argument]
    logger.info(f"using llm endpoint: {env.openai_compatible_base_url}")
    logger.info(f"using llm model: {env.llm_model}")

    llm.register_provider(
        "openai",
        scope="custom/",
        base_url=env.openai_compatible_base_url,
        api_key=env.openai_compatible_api_key,
    )

    model = llm.Model(f"custom/{env.llm_model}")

    # prepare the dataset example
    eventqa = naive_eventqa_example()
    chunker = SentenceAwareChunker()

    # prepare the agent
    agent = FullContextAgent(model=model, template=EventQATemplate())

    # chunk context and let agent memorize
    for chunk in chunker.chunk(eventqa.context):
        agent.memorize(chunk)

    # ask agent for question
    for question in eventqa.questions:
        resp = agent.query(question.text)
        print(f"{'=' * 40}\n{resp}\n{'=' * 40}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
