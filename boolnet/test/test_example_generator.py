import operator
import os.path
import numpy as np
from pytest import fixture
from numpy.testing import assert_array_equal as assert_array_equal
from bitpacking.packing import packed_type, packmat
import boolnet.bintools.operator_iterator as opit
from boolnet.bintools.example_generator import (
    packed_from_operator, PackedExampleGenerator)


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

        factory = opit.OpExampleIterFactory(
            ex_indices, Nb, binary_op[0], True)

        for i, (inp, tgt) in zip(indices, iter(factory)):
            assert i == inp
            expected_out = binary_op[1](int(i // upper), int(i % upper))
            if expected_out < 0:
                expected_out += upper
            assert expected_out == tgt

    def test_unary_operator_factory_include(self, unary_op, operand_width):
        Nb = int(operand_width)
        max_indices = 2**Nb
        Ne = np.random.randint(min(100, max_indices))
        indices = np.random.choice(max_indices, Ne, replace=False)
        indices = np.array(indices, dtype=np.uint64)

        factory = opit.OpExampleIterFactory(indices, Nb, unary_op[0], False)

        for i, (inp, tgt) in zip(indices, iter(factory)):
            assert i == inp
            expected_out = unary_op[1](i, max_indices)
            assert expected_out == tgt

    def test_unary_operator_factory_exclude(self, unary_op, operand_width):
        Nb = operand_width
        N = 2**Nb
        Ne = np.random.randint(min(100, N))
        ex_indices = np.sort(np.random.choice(N, Ne, replace=False))
        ex_indices = np.array(ex_indices, dtype=np.uint64)
        indices = (i for i in range(N) if i not in ex_indices)

        factory = opit.OpExampleIterFactory(
            ex_indices, Nb, unary_op[0], True)

        for i, (inp, tgt) in zip(indices, iter(factory)):
            assert i == inp
            expected_out = unary_op[1](i, N)
            assert expected_out == tgt


class TestExampleGenerator:
    op_map = {'add': opit.ADD, 'sub': opit.SUB, 'mul': opit.MUL}
    cache = dict()

    @fixture(params=['add4.npz', 'add8.npz', 'sub4.npz', 'sub8.npz',
                     'mul2.npz', 'mul3.npz', 'mul4.npz', 'mul6.npz',
                     'mul2f.npz', 'mul3f.npz', 'mul4f.npz', 'mul6f.npz'])
    def file_func_inst(self, request, test_location):
        return test_location, request.param

    @fixture(params=[True, False])
    def exclude(self, request):
        return request.param

    def load_file_func_instance(self, instance):
        location, name = instance
        fname = os.path.join(location, 'functions', name)
        if fname not in self.cache:
            with np.load(fname) as data:
                inp = data['input_matrix']
                tgt = data['target_matrix']
            self.cache[fname] = (inp, tgt)
        inp = np.array(self.cache[fname][0], copy=True)
        tgt = np.array(self.cache[fname][1], copy=True)
        return self.op_map[name[:3]], inp, tgt

    def build_generator_instance(self, instance, exclude):
        op, inp, tgt = self.load_file_func_instance(instance)

        Ne, Ni = inp.shape
        _, No = tgt.shape

        indices = np.random.choice(Ne, min(100, Ne-1), replace=False)
        indices = np.sort(indices)
        inp = inp[indices, :]
        tgt = tgt[indices, :]
        inp_p = packmat(inp)
        tgt_p = packmat(tgt)
        if exclude:
            indices = [i for i in range(Ne) if i not in indices]
        indices = np.array(indices, dtype=packed_type)

        factory = opit.OpExampleIterFactory(
            indices, Ni//2, op, Ne if exclude else 0)

        gen = PackedExampleGenerator(factory, No)
        return (gen, inp_p, tgt_p)

    @fixture(params=[0, np.random.rand(), 1])
    def block_fraction(self, request):
        return request.param

    def test_packed_generation(self, file_func_inst, exclude, block_fraction):
        generator_instance = self.build_generator_instance(
            file_func_inst, exclude)
        generator, expected_inp, expected_tgt = generator_instance

        actual_inp = np.zeros_like(expected_inp)
        actual_tgt = np.zeros_like(expected_tgt)
        Ni, cols = expected_inp.shape
        No, _ = expected_tgt.shape
        block_width = np.round(cols * block_fraction)
        block_width = int(np.clip(block_width, 1, cols))

        i = 0
        while generator:
            generator.next_examples(actual_inp[:, i: i + block_width],
                                    actual_tgt[:, i: i + block_width])
            i += block_width
        assert_array_equal(expected_inp, actual_inp)
        assert_array_equal(expected_tgt, actual_tgt)

    def test_packed_from_operator_inc(self, file_func_inst, exclude):
        op, inp, tgt = self.load_file_func_instance(file_func_inst)

        Ne, Ni = inp.shape
        _, No = tgt.shape

        indices = np.random.choice(Ne, min(100, Ne-1), replace=False)
        indices = np.sort(indices)
        indices = np.array(indices, dtype=packed_type)
        if exclude:
            expected_indices = [i for i in range(Ne) if i not in indices]
            expected_indices = np.array(expected_indices, dtype=packed_type)
        else:
            expected_indices = indices
        expected = packmat(np.hstack((
            inp[expected_indices, :],
            tgt[expected_indices, :])))

        actual = packed_from_operator(indices, Ni//2, No, op, exclude)

        assert_array_equal(expected, actual)
