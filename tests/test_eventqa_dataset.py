from tape_mem.dataset import load_eventqa_examples


def test_load_eventqa_examples_smoke():
    eventqa_examples = load_eventqa_examples()
    assert len(eventqa_examples) > 0
