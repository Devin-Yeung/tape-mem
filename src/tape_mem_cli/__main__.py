import click

from tape_mem.types.experiment import EventQAQueryResult, EventQAExperiment
import os

from tqdm import tqdm

from tape_mem.dataset.templates import EventQATemplate
from tape_mem.chunker import SentenceAwareChunker
from tape_mem.agents import FullContextAgent
from tape_mem.dataset.eventqa import naive_eventqa_example
from typing import List
from mirascope import llm

from .settings.env import Env
from loguru import logger

from rich.console import Console
from rich.table import Table


@click.command()
@click.option(
    "--model",
    "model_override",
    default=None,
    help="LLM model name, overrides env LLM_MODEL",
)
def main(model_override: str | None):
    env = Env()  # ty:ignore[missing-argument]

    if os.environ.get("HF_ENDPOINT") is None:
        # use mirror by default
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    logger.info(f"using hugging face endpoint: {os.environ['HF_ENDPOINT']}")

    # CLI --model takes precedence over env var
    llm_model = model_override or env.llm_model

    logger.info(f"using llm endpoint: {env.openai_compatible_base_url}")
    logger.info(f"using llm model: {llm_model}")

    llm.register_provider(
        "openai",
        scope="custom/",
        base_url=env.openai_compatible_base_url,
        api_key=env.openai_compatible_api_key,
    )

    model = llm.Model(f"custom/{llm_model}")

    # prepare the dataset example
    eventqa = naive_eventqa_example()
    chunker = SentenceAwareChunker()

    # prepare the agent
    agent = FullContextAgent(model=model, template=EventQATemplate())

    # chunk context and let agent memorize
    for chunk in chunker.chunk(eventqa.context):
        agent.memorize(chunk)

    console = Console()

    table = Table(title="Result", show_lines=True)

    table.add_column("Query")
    table.add_column("Answer")

    results: List[EventQAQueryResult] = []

    # ask agent for question
    for question in tqdm(eventqa.questions):
        resp = agent.query(question.text)
        table.add_row(question.text, resp.answer)
        results.append(
            EventQAQueryResult(
                question=question,
                response=resp,
            )
        )

    experiment = EventQAExperiment(results=results)
    json = experiment.to_json()

    with open("result.json", "w") as f:
        if isinstance(json, str):
            f.write(json)

    console.print(table)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
