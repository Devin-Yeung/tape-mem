from typing import List

import nltk
import tiktoken


def chunk_text_into_sentences(
    text: str, model_name: str = "gpt-4o-mini", chunk_size: int = 4096
) -> List[str]:
    """
    Split text into chunks of specified token size, preserving sentence boundaries.

    Args:
        text: The long text document to be split
        model_name: The tokenizer model name (default: gpt-4o-mini)
        chunk_size: Maximum number of tokens allowed per chunk

    Returns:
        List of text chunks, each within the specified token limit
    """
    # Ensure NLTK sentence tokenizer is available
    nltk.download("punkt", quiet=True)

    # Initialize tokenizer with fallback
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        # Use fallback model if specified model is not recognized
        encoding = tiktoken.encoding_for_model("gpt-4o-mini")

    # Split text into sentences
    sentences = nltk.sent_tokenize(text)

    text_chunks: List[str] = []
    current_chunk_sentences = []
    current_chunk_token_count = 0

    for sentence in sentences:
        # Count tokens in current sentence
        sentence_tokens = encoding.encode(sentence, allowed_special={"<|endoftext|>"})
        sentence_token_count = len(sentence_tokens)

        # Check if adding this sentence would exceed chunk size
        if current_chunk_token_count + sentence_token_count > chunk_size:
            # Finalize current chunk and start new one
            text_chunks.append(" ".join(current_chunk_sentences))
            current_chunk_sentences = [sentence]
            current_chunk_token_count = sentence_token_count
        else:
            # Add sentence to current chunk
            current_chunk_sentences.append(sentence)
            current_chunk_token_count += sentence_token_count

    # Add final chunk if it contains any sentences
    if current_chunk_sentences:
        text_chunks.append(" ".join(current_chunk_sentences))

    return text_chunks
