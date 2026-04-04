from tape_mem.evaluation import CandidateMatchEvaluator
from tape_mem.types import Evaluator


def test_candidate_match_evaluator_implements_protocol():
    evaluator = CandidateMatchEvaluator()
    assert isinstance(evaluator, Evaluator)


def test_candidate_match_evaluator_matches_case_and_whitespace_variants():
    evaluator = CandidateMatchEvaluator()

    result = evaluator.evaluate("  The   Answer  ", ("the answer", "something else"))

    assert result.matched is True
    assert result.normalized_prediction == "the answer"
    assert result.normalized_candidates == ("the answer", "something else")


def test_candidate_match_evaluator_does_not_strip_punctuation():
    evaluator = CandidateMatchEvaluator()

    result = evaluator.evaluate("Answer!", ("answer",))

    assert result.matched is False
    assert result.normalized_prediction == "answer!"
