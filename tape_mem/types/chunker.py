from typing import Protocol, List, runtime_checkable


@runtime_checkable
class Chunker(Protocol):
    """
    A text chunking strategy that splits long context into smaller, manageable pieces.

    Implementors decide how to segment text — by tokens, sentences, paragraphs, or
    semantic boundaries — while respecting the contract that each chunk fits within
    model context limits.
    """

    def chunk(self, context: str) -> List[str]:
        """
        Split the given context into a list of chunks.

        Args:
            context: The text to be split into chunks.

        Returns:
            A list of text segments, each representing a chunk of the input.
        """
        ...
