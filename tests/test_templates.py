import pytest

from tape_mem.dataset.templates import get_template


def test_example_template_snapshot(snapshot):
    actual = get_template("ruler_qa", "memorize", "rag_agent")
    assert actual == snapshot


@pytest.mark.parametrize(
    "ds",
    [
        "ruler_qa",
        "icl_*",
        "infbench_*sum",
        "eventqa_*",
        "recsys_*redial",
        "longmemeval_",
        "factconsolidation_*",
        "detective_*qa",
    ],
)
@pytest.mark.parametrize(
    "agent", ["rag_agent", "long_context_agent", "agentic_memory_agent"]
)
@pytest.mark.parametrize("tpl_type", ["system", "memorize", "query"])
def test_all_datasets_have_all_template_types(ds, agent, tpl_type):
    result = get_template(ds, tpl_type, agent)
    assert isinstance(result, str)
