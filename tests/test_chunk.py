import pytest

import nltk

from tape_mem.datasets.utils.chunk import chunk_text_into_sentences


@pytest.fixture(scope="module", autouse=True)
def download_nltk_data():
    nltk.download("punkt_tab", quiet=True)


class TestChunkTextIntoSentences:
    def test_empty_text(self):
        result = chunk_text_into_sentences("")
        assert result == []

    def test_single_sentence(self):
        text = "This is a single sentence."
        result = chunk_text_into_sentences(text, chunk_size=100)
        assert len(result) == 1
        assert result[0] == text

    def test_multiple_sentences_chunked(self):
        text = " ".join(["Sentence {}.".format(i) for i in range(50)])
        result = chunk_text_into_sentences(text, chunk_size=50)
        assert len(result) > 1

    def test_unknown_model_fallback(self):
        text = "First sentence. Second sentence."
        result = chunk_text_into_sentences(text, model_name="unknown-model")
        assert len(result) >= 1
