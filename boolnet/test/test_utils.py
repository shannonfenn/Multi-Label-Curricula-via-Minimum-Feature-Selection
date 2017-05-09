import pytest
import boolnet.utils as utils


@pytest.fixture
def harness(request):
    ranking_with_ties = [3, 6, 3, 1, 0, 7, 7, 3, 1]
    valid_rankings = [
        [3, 6, 4, 1, 0, 7, 8, 5, 2],
        [3, 6, 4, 2, 0, 7, 8, 5, 1],
        [3, 6, 5, 1, 0, 7, 8, 4, 2],
        [3, 6, 5, 2, 0, 7, 8, 4, 1],
        [4, 6, 3, 1, 0, 7, 8, 5, 2],
        [4, 6, 3, 2, 0, 7, 8, 5, 1],
        [4, 6, 5, 1, 0, 7, 8, 3, 2],
        [4, 6, 5, 2, 0, 7, 8, 3, 1],
        [5, 6, 3, 1, 0, 7, 8, 4, 2],
        [5, 6, 3, 2, 0, 7, 8, 4, 1],
        [5, 6, 4, 1, 0, 7, 8, 3, 2],
        [5, 6, 4, 2, 0, 7, 8, 3, 1],
        [3, 6, 4, 1, 0, 8, 7, 5, 2],
        [3, 6, 4, 2, 0, 8, 7, 5, 1],
        [3, 6, 5, 1, 0, 8, 7, 4, 2],
        [3, 6, 5, 2, 0, 8, 7, 4, 1],
        [4, 6, 3, 1, 0, 8, 7, 5, 2],
        [4, 6, 3, 2, 0, 8, 7, 5, 1],
        [4, 6, 5, 1, 0, 8, 7, 3, 2],
        [4, 6, 5, 2, 0, 8, 7, 3, 1],
        [5, 6, 3, 1, 0, 8, 7, 4, 2],
        [5, 6, 3, 2, 0, 8, 7, 4, 1],
        [5, 6, 4, 1, 0, 8, 7, 3, 2],
        [5, 6, 4, 2, 0, 8, 7, 3, 1]]
    valid_orderings = [
        [4, 3, 8, 0, 2, 7, 1, 5, 6],
        [4, 3, 8, 0, 2, 7, 1, 6, 5],
        [4, 3, 8, 0, 7, 2, 1, 5, 6],
        [4, 3, 8, 0, 7, 2, 1, 6, 5],
        [4, 3, 8, 2, 0, 7, 1, 5, 6],
        [4, 3, 8, 2, 0, 7, 1, 6, 5],
        [4, 3, 8, 2, 7, 0, 1, 5, 6],
        [4, 3, 8, 2, 7, 0, 1, 6, 5],
        [4, 3, 8, 7, 0, 2, 1, 5, 6],
        [4, 3, 8, 7, 0, 2, 1, 6, 5],
        [4, 3, 8, 7, 2, 0, 1, 5, 6],
        [4, 3, 8, 7, 2, 0, 1, 6, 5],
        [4, 8, 3, 0, 2, 7, 1, 5, 6],
        [4, 8, 3, 0, 2, 7, 1, 6, 5],
        [4, 8, 3, 0, 7, 2, 1, 5, 6],
        [4, 8, 3, 0, 7, 2, 1, 6, 5],
        [4, 8, 3, 2, 0, 7, 1, 5, 6],
        [4, 8, 3, 2, 0, 7, 1, 6, 5],
        [4, 8, 3, 2, 7, 0, 1, 5, 6],
        [4, 8, 3, 2, 7, 0, 1, 6, 5],
        [4, 8, 3, 7, 0, 2, 1, 5, 6],
        [4, 8, 3, 7, 0, 2, 1, 6, 5],
        [4, 8, 3, 7, 2, 0, 1, 5, 6],
        [4, 8, 3, 7, 2, 0, 1, 6, 5]]
    return ranking_with_ties, valid_rankings, valid_orderings


@pytest.mark.parametrize('execution_number', range(10))
def test_rank_with_ties_broken(harness, execution_number):
    ranking_with_ties, all_expected, _ = harness
    actual = utils.rank_with_ties_broken(ranking_with_ties)
    assert actual.tolist() in all_expected


@pytest.mark.parametrize('execution_number', range(10))
def test_order_from_rank(harness, execution_number):
    ranking_with_ties, _, all_expected = harness
    actual = utils.order_from_rank(ranking_with_ties)
    assert actual.tolist() in all_expected
