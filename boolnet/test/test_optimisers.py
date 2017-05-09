from boolnet.optimisers import SA, stepped_exp_decrease, geometric
from math import exp
import numpy as np
import operator as op
import random


class TestSA:
    geometric_series = [(1, 0.5, 5, list(1*0.5**i for i in range(5))),
                        (5, 0.95, 20, list(5*0.95**i for i in range(20)))]

    def test_geometric(self):
        for series in self.geometric_series:
            expected = series[3]
            actual = [round(x, 10) for x in geometric(series[0], series[1], series[2])]
            assert np.allclose(expected, actual)

    def test_stepped_exp_decrease(self):
        for series in self.geometric_series:
            repeats = random.randrange(50)
            expected = []
            for t in series[3]:
                for r in range(repeats):
                    expected.append(t)
            actual = stepped_exp_decrease(series[0], series[1], series[2], repeats)
            actual = [round(x, 10) for x in actual]
            assert np.allclose(expected, actual)

    def test_accept(self):
        annealer = SA()
        # inject default
        annealer.is_better = op.lt
        assert annealer.accept(1.0, 0.5, 100)
        assert annealer.accept(1.0, 1.0, 100)
        assert not annealer.accept(0.0, 0.5, 0)

        random.seed(103)
        ran = [random.random() for i in range(10)]

        for d in [0.4, 0.8, 1.2, 1.6, 2.0]:
            for t in [0.2, 0.4, 0.6, 0.8, 1.0]:
                random.seed(103)
                for r in ran:
                    assert annealer.accept(0, d, t) == (r < exp(-d/t))
