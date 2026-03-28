from tape_mem.types.conversation import Turn


def test_turn_serialization_without_metadata(snapshot):
    """Test Turn serializes correctly when metadata is None."""
    turn = Turn(user="Hello", agent="Hi there!")
    assert turn.to_json() == snapshot


def test_turn_serialization_with_metadata(snapshot):
    """Test Turn serializes correctly with a simple metadata dict."""
    metadata = {"memorize": False}
    turn = Turn(user="Hello", agent="Hi there!", metadata=metadata)
    assert turn.to_json() == snapshot
