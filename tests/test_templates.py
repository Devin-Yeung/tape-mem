import pytest

from tape_mem.datasets.templates import (
    SYSTEM_MESSAGE,
    BASE_TEMPLATES,
    AGENT_TYPE_MAPPING,
    DATASET_MAPPING,
    get_template,
)


def test_system_message_is_nonempty_string():
    assert isinstance(SYSTEM_MESSAGE, str)
    assert len(SYSTEM_MESSAGE) > 0


def test_base_templates_is_dict():
    assert isinstance(BASE_TEMPLATES, dict)
    assert len(BASE_TEMPLATES) > 0


def test_base_templates_has_expected_keys():
    expected_keys = {
        "ruler_qa",
        "longmemeval",
        "eventqa",
        "in_context_learning",
        "recsys_redial",
        "infbench_sum",
        "detective_qa",
        "factconsolidation",
    }
    assert set(BASE_TEMPLATES.keys()) == expected_keys


def test_each_base_template_has_system_memorize_query():
    for name, template in BASE_TEMPLATES.items():
        assert "system" in template, f"{name} missing 'system'"
        assert "memorize" in template, f"{name} missing 'memorize'"
        assert "query" in template, f"{name} missing 'query'"


def test_query_templates_are_dicts_with_agent_keys():
    expected_agents = {"long_context_agent", "rag_agent", "agentic_memory_agent"}
    for name, template in BASE_TEMPLATES.items():
        assert isinstance(template["query"], dict), f"{name} query not a dict"
        assert set(template["query"].keys()) == expected_agents, (
            f"{name} query keys mismatch"
        )


def test_agent_type_mapping_has_expected_values():
    expected_values = {"long_context_agent", "rag_agent", "agentic_memory_agent"}
    assert set(AGENT_TYPE_MAPPING.values()) == expected_values


def test_dataset_mapping_values_match_base_templates_keys():
    assert set(DATASET_MAPPING.values()) == set(BASE_TEMPLATES.keys())


# ==============================================================================
# get_template
# ==============================================================================


@pytest.mark.parametrize(
    "agent", ["rag_agent", "long_context_agent", "agentic_memory_agent"]
)
def test_system_template(agent):
    # system template is a plain string, not agent-specific
    assert get_template("ruler_hqa", "system", agent) == SYSTEM_MESSAGE


@pytest.mark.parametrize(
    "agent", ["rag_agent", "long_context_agent", "agentic_memory_agent"]
)
def test_memorize_template(agent):
    # memorize template is a plain string, not agent-specific
    result = get_template("ruler_hqa", "memorize", agent)
    assert isinstance(result, str)
    assert "{context}" in result


def test_query_template_rag_agent():
    result = get_template("ruler_hqa", "query", "rag_agent")
    assert isinstance(result, str)
    assert "Answer the question" in result


def test_query_template_long_context_agent():
    result = get_template("ruler_hqa", "query", "long_context_agent")
    assert isinstance(result, str)
    assert "memorized documents" in result


def test_query_template_agentic_memory():
    result = get_template("ruler_hqa", "query", "agentic_memory_agent")
    assert isinstance(result, str)
    assert "Archival Memory" in result


@pytest.mark.parametrize(
    "ds",
    [
        "ruler_hqa",
        "icl_something",
        "infbench_sum",
        "eventqa_something",
        "recsys_redial",
        "longmemeval_something",
        "factconsolidation_something",
        "detective_qa",
    ],
)
@pytest.mark.parametrize(
    "agent", ["rag_agent", "long_context_agent", "agentic_memory_agent"]
)
@pytest.mark.parametrize("tpl_type", ["system", "memorize", "query"])
def test_all_datasets_have_all_template_types(ds, agent, tpl_type):
    result = get_template(ds, tpl_type, agent)
    assert isinstance(result, str)
