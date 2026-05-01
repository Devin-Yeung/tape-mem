from tape_mem.dataset import load_longmemeval_examples


def test_load_longmemeval_examples_smoke(snapshot):
    examples = load_longmemeval_examples()
    example_ids = [x.example_id for x in examples]
    assert example_ids == snapshot


def test_load_longmemeval_one_session(snapshot):
    """Snapshot the first session from the first LongMemEval example."""
    examples = load_longmemeval_examples()
    example = examples[0]
    session = example.sessions[0]

    # Snapshot the session header and first message
    assert session.chat_time == snapshot
    assert session.messages[0].content[:50] == snapshot


def test_load_longmemeval_questions(snapshot):
    examples = load_longmemeval_examples()
    example = examples[0]

    # serialize the questions
    questions = [q.to_dict() for q in example.questions]

    assert questions == snapshot
