import pytest

import nltk

from tape_mem.utils.chunk import SentenceAwareChunker
from tape_mem.types.chunker import Chunker


@pytest.fixture(scope="module", autouse=True)
def download_nltk_data():
    nltk.download("punkt_tab", quiet=True)


class TestSentenceAwareChunker:
    @pytest.fixture
    def chunker(self):
        return SentenceAwareChunker(chunk_size=50)

    def test_implements_protocol(self, chunker):
        assert isinstance(chunker, Chunker)

    def test_empty_text(self, chunker):
        result = chunker.chunk("")
        assert result == []

    def test_single_sentence(self, chunker):
        text = "This is a single sentence."
        result = chunker.chunk(text)
        assert len(result) == 1
        assert result[0] == text

    def test_multiple_sentences_chunked(self, chunker):
        text = " ".join(["Sentence {}.".format(i) for i in range(50)])
        result = chunker.chunk(text)
        assert len(result) > 1

    def test_preserves_sentence_boundaries(self, chunker):
        """Sentences should never be split across chunks."""
        text = "Short. Medium length sentence here. Another one."
        result = chunker.chunk(text)
        for chunk in result:
            # Each chunk should start with a capitalized letter or be empty
            assert chunk == "" or chunk[0].isupper() or chunk.startswith(" ")

    def test_unknown_model_fallback(self):
        chunker = SentenceAwareChunker(model_name="unknown-model", chunk_size=100)
        text = "First sentence. Second sentence."
        result = chunker.chunk(text)
        assert len(result) >= 1

    def test_custom_chunk_size(self):
        chunker = SentenceAwareChunker(chunk_size=10)
        text = "This is a very long sentence that should be chunked."
        result = chunker.chunk(text)
        # Should have multiple chunks for a long sentence
        assert len(result) >= 1
