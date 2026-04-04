from typing import List

import nltk
import tiktoken


class SentenceAwareChunker:
    """
    A chunker that splits text into token-bounded chunks while preserving sentence boundaries.

    This implementation wraps the sentence-aware chunking logic behind the Chunker
    protocol, ensuring no sentence is split across chunks.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        chunk_size: int = 4096,
    ) -> None:
        """
        Initialize the chunker with tokenizer and size configuration.

        Args:
            model_name: The tokenizer model name for token counting (default: gpt-4o-mini)
            chunk_size: Maximum number of tokens allowed per chunker (default: 4096)
        """
        self._model_name = model_name
        self._chunk_size = chunk_size
        self._encoding = self._init_encoding()
        # Ensure NLTK sentence tokenizer is available at init time
        nltk.download("punkt", quiet=True)

    def _init_encoding(self):
        """Initialize the tiktoken encoder with fallback support."""
        try:
            return tiktoken.encoding_for_model(self._model_name)
        except KeyError:
            return tiktoken.encoding_for_model("gpt-4o-mini")

    def chunk(self, context: str) -> List[str]:
        """
        Chunk the given context into a list of sentence-bounded chunks.

        Args:
            context: The long text document to be split

        Returns:
            List of text chunks, each within the specified token limit
        """
        # Split text into sentences
        sentences = nltk.sent_tokenize(context)

        text_chunks: List[str] = []
        current_chunk_sentences: List[str] = []
        current_chunk_token_count = 0

        for sentence in sentences:
            sentence_tokens = self._encoding.encode(
                sentence, allowed_special={"<|endoftext|>"}
            )
            sentence_token_count = len(sentence_tokens)

            if current_chunk_token_count + sentence_token_count > self._chunk_size:
                # Finalize current chunker and start new one
                text_chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = [sentence]
                current_chunk_token_count = sentence_token_count
            else:
                current_chunk_sentences.append(sentence)
                current_chunk_token_count += sentence_token_count

        # Add final chunker if it contains any sentences
        if current_chunk_sentences:
            text_chunks.append(" ".join(current_chunk_sentences))

        # text chunker should never return empty strings
        text_chunks = [chunk for chunk in text_chunks if chunk.strip()]
        return text_chunks
