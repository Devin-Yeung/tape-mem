from tape_mem.dataset import load_eventqa_examples


def test_load_eventqa_examples_smoke(snapshot):
    eventqa_examples = load_eventqa_examples()
    example_ids = [x.example_id for x in eventqa_examples]
    assert example_ids == snapshot
