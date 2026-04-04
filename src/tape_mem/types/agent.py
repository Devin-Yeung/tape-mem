from typing import Protocol, runtime_checkable


@runtime_checkable
class Agent(Protocol):
    """A protocol defining the interface for a memory-augmented agent.

    The agent maintains an internal memory store that can be populated via
    memorize(), selectively cleared via forget(), and queried via query().

    Implementations are expected to be stateful — calls to memorize() should
    accumulate within the agent's memory across invocations, until explicitly
    forgotten or the agent is re-initialized.
    """

    def memorize(self, chunk: str) -> None:
        """Add a piece of information to the agent's memory store.

        The agent should append or index the chunk so it can be retrieved
        later via query(). Implementations may choose to deduplicate or
        merge semantically similar chunks.
        """
        ...

    def forget(self, chunk: str) -> None:
        """Remove a previously memorized chunk from the agent's memory.

        If the exact chunk is not found, implementations should silently
        succeed (no-op) rather than raise an error. Not all implementations
        may support precise forgetting — in that case, this method may raise
        NotImplementedError.
        """
        ...

    def query(self, question: str) -> str:
        """Ask the agent a question using its accumulated memory as context.

        The agent should answer based on information previously stored via
        memorize(). The question itself should NOT be added to memory —
        only the answer (if any) should be stored if the implementation
        chooses to do so.

        Args:
            question: The question to ask the agent.

        Returns:
            The agent's answer as a string. If no relevant information is
            found in memory, implementations may return an empty string or
            a message indicating the information is unknown.
        """
        ...
