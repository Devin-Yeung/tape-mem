from tape_mem.types.provider import ProviderConfig
import os
import random
from typing import Literal
from typing import List
import questionary

import click
from loguru import logger
from mirascope import llm
from tqdm import tqdm

from tape_mem.agents import FullContextAgent
from tape_mem.agents.rag import RagAgent
from tape_mem.agents.tape import TapeAgent
from tape_mem.chunker import SentenceAwareChunker
from tape_mem.dataset import load_eventqa_examples
from tape_mem.dataset import load_longmemeval_examples
from tape_mem.dataset.templates import EventQATemplate
from tape_mem.dataset.templates import LongMemEvalTemplate
from tape_mem.types.experiment import LongMemEvalExperiment
from tape_mem.types.experiment import LongMemEvalQueryResult
from .settings.env import Env

EVENTQA_VARIANTS = [
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

LONGMEMEVAL_VARIANTS = [
    "longmemeval_s*_0",
    "longmemeval_s*_1",
    "longmemeval_s*_2",
    "longmemeval_s*_3",
    "longmemeval_s*_4",
]


@click.command()
@click.option(
    "--model",
    "model_override",
    default=None,
    help="LLM model name, overrides env LLM_MODEL",
)
@click.option(
    "--dataset",
    "dataset_kind",
    type=click.Choice(["eventqa", "longmemeval"], case_sensitive=False),
    default="eventqa",
    show_default=True,
    help="Dataset to use for the experiment",
)
@click.option(
    "--variant",
    type=str,
    default=None,
    help="Experiment variant to run (filtered by dataset)",
)
@click.option(
    "--agent",
    "agent_kind",
    type=click.Choice(["full", "rag", "tape"], case_sensitive=False),
    default="full",
    show_default=True,
    help="Agent implementation to run",
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
    default=42,
    show_default=True,
)
def main(
    model_override: str | None,
    dataset_kind: Literal["eventqa", "longmemeval"],
    variant: str | None,
    agent_kind: Literal["full", "rag", "tape"],
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

    # prepare variants based on dataset
    variants = EVENTQA_VARIANTS if dataset_kind == "eventqa" else LONGMEMEVAL_VARIANTS

    if variant is None:
        variant = questionary.select(
            f"Select a {dataset_kind} variant to run:",
            choices=variants,
        ).ask()

    logger.info(f"running experiment on {variant}")

    # load dataset based on type
    if dataset_kind == "eventqa":
        examples = load_eventqa_examples()
        examples = [e for e in examples if e.example_id == variant]
        template = EventQATemplate()
        use_conversation = False
    else:
        examples = load_longmemeval_examples()
        examples = [e for e in examples if e.example_id == variant]
        template = LongMemEvalTemplate()
        use_conversation = True

    # prepare the agent
    match agent_kind:
        case "full":
            agent = FullContextAgent(model=model, template=template)
        case "rag":
            agent = RagAgent(model=model, template=template)
        case "tape":
            provider = ProviderConfig(
                model=llm_model,
                base_url=env.openai_compatible_base_url,
                api_key=env.openai_compatible_api_key,
            )
            agent = TapeAgent(provider, template=template)

    logger.info(f"using agent: {agent.__class__.__name__}")

    for subset_idx, subset in enumerate(tqdm(examples)):
        logger.info(f"processing {subset_idx} th subset of {variant}")

        # memorize context based on dataset type
        if use_conversation:
            results: List[LongMemEvalQueryResult] = []
            agent.memorize_conversation(list(subset.sessions))
        else:
            results = []
            for chunk in chunker.chunk(subset.context):
                agent.memorize(chunk)

        # sample questions
        n_selected = int(len(subset.questions) * (question_percent / 100.0))
        logger.info(f"selected {n_selected} questions")
        rng = random.Random(seed)
        selected = rng.sample(subset.questions, n_selected)

        # ask agent for question
        for i, question in enumerate(tqdm(selected)):
            logger.debug(f"processing question {i}")
            resp = agent.query(question.question_text)
            results.append(
                LongMemEvalQueryResult(
                    question=question,
                    response=resp,
                )
            )

        experiment = LongMemEvalExperiment(results=results)
        json = experiment.to_json()

        with open(f"{variant}_result.json", "w") as f:
            if isinstance(json, str):
                f.write(json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
