from tape_mem.dataset import load_longmemeval_examples


def test_load_longmemeval_examples_smoke(snapshot):
    examples = load_longmemeval_examples()
    example_ids = [x.example_id for x in examples]
    assert example_ids == snapshot


def test_load_longmemeval_one_session(snapshot):
    """Snapshot the first session from the first LongMemEval example."""
    examples = load_longmemeval_examples()
    first_example = examples[0]
    first_session = first_example.sessions[0]

    # Snapshot the session header and first message
    assert first_session.chat_time == snapshot
    assert first_session.messages[0].content[:50] == snapshot
