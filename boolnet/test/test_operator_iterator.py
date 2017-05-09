import operator
import numpy as np
from pytest import fixture
from numpy.testing import assert_array_equal as assert_array_equal
import boolnet.bintools.operator_iterator as opit


class TestIterators:
    @fixture(params=[
        opit.ZeroIncIterator,
        opit.UnaryANDIncIterator,
        opit.UnaryORIncIterator,
        opit.ANDIncIterator,
        opit.ORIncIterator,
        opit.AddIncIterator,
        opit.SubIncIterator,
        opit.MulIncIterator])
    def include_iterator(self, request):
        return request.param

    @fixture(params=[
        opit.ZeroExcIterator,
        opit.UnaryANDExcIterator,
        opit.UnaryORExcIterator,
        opit.ANDExcIterator,
        opit.ORExcIterator,
        opit.AddExcIterator,
        opit.SubExcIterator,
        opit.MulExcIterator])
    def exclude_iterator(self, request):
        return request.param

    @fixture(params=np.random.randint(2, 100, 5))
    def index_harness(self, request):
        max_index = request.param
        num_indices = np.random.randint(max_index-1, max_index)
        in_indices = np.random.choice(max_index, size=num_indices,
                                      replace=False)
        in_indices = np.array(np.sort(in_indices), dtype=np.uint64)
        ex_indices = np.array([i for i in range(max_index)
                               if i not in in_indices])
        return (in_indices, ex_indices, max_index)

    def test_include_indices(self, include_iterator, index_harness):
        indices, _, _ = index_harness
        expected = np.array(indices, copy=True)
        actual = np.array(list(include_iterator(indices, 2)))[:, 0]

        assert_array_equal(expected, actual)

    def test_exclude_indices(self, exclude_iterator, index_harness):
        indices, expected, max_index = index_harness
        actual = np.array(list(exclude_iterator(indices, 2, max_index)))[:, 0]

        assert_array_equal(expected, actual)


class TestExampleIteratorFactory:
    @fixture(params=[
        (opit.AND, operator.__and__),
        (opit.OR, operator.__or__),
        (opit.ADD, operator.add),
        (opit.SUB, operator.sub),
        (opit.MUL, operator.mul)
    ])
    def binary_op(self, request):
        return request.param

    @fixture(params=[
        (opit.ZERO, lambda x, m: 0),
        (opit.UNARY_OR, lambda x, m: int(x > 0)),
        (opit.UNARY_AND, lambda x, m: int(x == m - 1))
    ])
    def unary_op(self, request):
        return request.param

    @fixture(params=list(range(1, 7)))
    def operand_width(self, request):
        return request.param

    def test_operator_factory_include(self, binary_op, operand_width):
        operator_id, operator_function = binary_op
        Nb = int(operand_width)
        upper = 2**Nb
        max_indices = 2**(2*Nb)
        Ne = np.random.randint(min(100, max_indices))
        indices = np.random.choice(max_indices, Ne, replace=False)
        indices = np.array(indices, dtype=np.uint64)

        factory = opit.OpExampleIterFactory(indices, Nb, operator_id, False)

        for i, (inp, tgt) in zip(indices, iter(factory)):
            assert i == inp
            expected_out = operator_function(int(i // upper), int(i % upper))
            if expected_out < 0:
                expected_out += upper
            assert expected_out == tgt

    def test_operator_factory_exclude(self, binary_op, operand_width):
        Nb = operand_width
        upper = 2**Nb
        N = 2**(2*Nb)
        Ne = np.random.randint(min(100, N))
        ex_indices = np.random.choice(N, Ne, replace=False)
        ex_indices = np.array(ex_indices, dtype=np.uint64)
        indices = (i for i in range(N) if i not in ex_indices)

        factory = opit.OpExampleIterFactory(ex_indices, Nb, binary_op[0], True)

        for i, (inp, tgt) in zip(indices, iter(factory)):
            assert i == inp
            expected_out = binary_op[1](int(i // upper), int(i % upper))
            if expected_out < 0:
                expected_out += upper
            assert expected_out == tgt

    def test_unary_operator_factory_include(self, unary_op, operand_width):
        Nb = int(operand_width)
        N = 2**Nb
        Ne = np.random.randint(min(100, N))
        indices = np.random.choice(N, Ne, replace=False)
        indices = np.array(indices, dtype=np.uint64)

        factory = opit.OpExampleIterFactory(indices, Nb, unary_op[0], False)

        for i, (inp, tgt) in zip(indices, iter(factory)):
            assert i == inp
            expected_out = unary_op[1](i, N)
            assert expected_out == tgt

    def test_unary_operator_factory_exclude(self, unary_op, operand_width):
        Nb = operand_width
        N = 2**Nb
        Ne = np.random.randint(min(100, N))
        ex_indices = np.sort(np.random.choice(N, Ne, replace=False))
        ex_indices = np.array(ex_indices, dtype=np.uint64)
        indices = (i for i in range(N) if i not in ex_indices)

        factory = opit.OpExampleIterFactory(ex_indices, Nb, unary_op[0], True)

        for i, (inp, tgt) in zip(indices, iter(factory)):
            assert i == inp
            expected_out = unary_op[1](i, N)
            assert expected_out == tgt
