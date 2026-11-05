from eval.run_eval import Case, score

from tests.conftest import make_hit


def test_scores_zero_instead_of_crashing_without_answerable_questions():
    results = [(Case(question="Who won the World Cup?", page=None), [make_hit("Unrelated.", similarity=0.05)])]

    report = score(results, min_similarity=0.15)

    assert report.hit_at_1 == 0
    assert report.mrr == 0
    assert report.answered_answerable == 0
    assert report.refused_unanswerable == 1
