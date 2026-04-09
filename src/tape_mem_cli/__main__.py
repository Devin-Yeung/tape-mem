import os
from typing import Literal
from typing import List

import click
from loguru import logger
from mirascope import llm
from tqdm import tqdm

from tape_mem.agents import FullContextAgent
from tape_mem.chunker import SentenceAwareChunker
from tape_mem.dataset import load_eventqa_examples
from tape_mem.dataset.templates import EventQATemplate
from tape_mem.types.experiment import EventQAExperiment
from tape_mem.types.experiment import EventQAQueryResult
import questionary
from .settings.env import Env

VARIANTS = [
    "eventqa_full_0",
    "eventqa_full_1",
    "eventqa_full_2",
    "eventqa_full_3",
    "eventqa_full_4",
    "eventqa_65536_0",
    "eventqa_65536_1",
    "eventqa_65536_2",
    "eventqa_65536_3",
    "eventqa_65536_4",
    "eventqa_131072_0",
    "eventqa_131072_1",
    "eventqa_131072_2",
    "eventqa_131072_3",
    "eventqa_131072_4",
]


@click.command()
@click.option(
    "--model",
    "model_override",
    default=None,
    help="LLM model name, overrides env LLM_MODEL",
)
@click.option(
    "--variant",
    type=click.Choice(VARIANTS, case_sensitive=False),
    default=None,
    help="Experiment variant to run",
)
@click.option(
    "--question-percent",
    type=click.FloatRange(0.0, 100.0),
    default=10.0,
    show_default=True,
    help="Percent of the eligible questions to run for each subset",
)
@click.option(
    "--seed",
    default="default",
    show_default=True,
)
def main(
    model_override: str | None,
    variant: Literal[
        "eventqa_full_0",
        "eventqa_full_1",
        "eventqa_full_2",
        "eventqa_full_3",
        "eventqa_full_4",
        "eventqa_65536_0",
        "eventqa_65536_1",
        "eventqa_65536_2",
        "eventqa_65536_3",
        "eventqa_65536_4",
        "eventqa_131072_0",
        "eventqa_131072_1",
        "eventqa_131072_2",
        "eventqa_131072_3",
        "eventqa_131072_4",
    ]
    | None,
    question_percent: float,
    seed: int,
) -> int:
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

    # setup chunker
    chunker = SentenceAwareChunker()
    logger.info(f"using chunker: {chunker}")

    # prepare the dataset example
    eventqa = load_eventqa_examples()

    if variant is None:
        variant = questionary.select(
            "Select a variant to run:",
            choices=VARIANTS,
        ).ask()

    logger.info(f"running experiment on {variant}")
    eventqa = [e for e in eventqa if e.example_id == variant]

    # prepare the agent
    agent = FullContextAgent(model=model, template=EventQATemplate())

    for subset_idx, subset in enumerate(tqdm(eventqa)):
        logger.info(f"processing {subset_idx} th subset of {variant}")
        results: List[EventQAQueryResult] = []
        # chunk context and let agent memorize
        for chunk in chunker.chunk(subset.context):
            agent.memorize(chunk)

        # ask agent for question
        for i, question in enumerate(tqdm(subset.questions)):
            logger.debug(f"processing question {i}")
            resp = agent.query(question.text)
            results.append(
                EventQAQueryResult(
                    question=question,
                    response=resp,
                )
            )

        experiment = EventQAExperiment(results=results)
        json = experiment.to_json()

        with open(f"{variant}-{subset_idx}-result.json", "w") as f:
            if isinstance(json, str):
                f.write(json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
