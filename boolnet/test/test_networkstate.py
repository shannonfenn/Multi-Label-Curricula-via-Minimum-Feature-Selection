import yaml
import glob
import numpy as np
from copy import copy, deepcopy
from collections import namedtuple
from pytest import mark, raises, fixture

import bitpacking.packing as pk
from bitpacking.packing import PACKED_SIZE_PY as PACKED_SIZE

from boolnet.utils import PackedMatrix, partition_packed
from boolnet.bintools.functions import (
    function_name, all_functions, function_from_name)
from boolnet.bintools.operator_iterator import (
    ZERO, AND, OR, UNARY_AND, UNARY_OR, ADD, SUB, MUL,
    OpExampleIterFactory)
from boolnet.bintools.example_generator import (packed_from_operator,
                                                PackedExampleGenerator)
from boolnet.network.networkstate import BNState, state_from_operator


TEST_NETWORKS = glob.glob('boolnet/test/networks/*.yaml')


operator_map = {
    'and': AND, 'or': OR,
    'add': ADD, 'sub': SUB, 'mul': MUL,
    'zero': ZERO, 'unary_and': UNARY_AND, 'unary_or': UNARY_OR
}


# ############# General helpers ################# #
def to_binary(value, num_bits):
    return [int(i) for i in '{:0{w}b}'.format(value, w=num_bits)][::-1]


def all_possible_inputs(num_bits):
    return pk.packmat(np.array(
        [to_binary(i, num_bits) for i in range(2**num_bits)],
        dtype=np.uint8))


def packed_zeros(shape):
    return pk.packmat(np.zeros(shape, dtype=np.uint8))


# ############ Helpers for fixtures ############# #
def harnesses_with_property(bool_property_name):
    for name in TEST_NETWORKS:
        with open(name) as f:
            test = yaml.safe_load(f)
            if test[bool_property_name]:
                yield name


HARNESS_CACHE = dict()


def harness_to_fixture(fname):
    if fname not in HARNESS_CACHE:
        with open(fname) as stream:
            test = yaml.safe_load(stream)
            HARNESS_CACHE[fname] = test

    test = deepcopy(HARNESS_CACHE[fname])

    Ni = test['Ni']
    No = test['No']
    gates = np.array(test['gates'], np.uintp)
    samples = np.array(test['samples'], np.uintp)

    # add non-existant sub-dictionaries
    test['input matrix'] = {}
    test['output matrix'] = {}

    target = np.array(test['target matrix']['full'], dtype=np.uint8)
    activation = np.array(test['activation matrix']['full'], dtype=np.uint8)
    error = np.array(test['error matrix']['full'], dtype=np.uint8)
    inputs = activation[:, :Ni]
    test['target matrix']['full'] = target
    test['input matrix']['full'] = inputs
    test['output matrix']['full'] = activation[:, -No:]
    test['activation matrix']['full'] = activation
    test['error matrix']['full'] = error

    # add sample version of expectations to test
    test['target matrix']['sample'] = target[samples]
    test['input matrix']['sample'] = inputs[samples]
    test['output matrix']['sample'] = activation[samples][:, -No:]
    test['activation matrix']['sample'] = activation[samples]
    test['error matrix']['sample'] = error[samples]

    # Test sample version
    samples_t = np.array([i for i in range(inputs.shape[0])
                          if i not in samples], dtype=np.uintp)

    target_t = target[samples_t]
    inputs_t = inputs[samples_t]
    activation_t = activation[samples_t]
    # add test version of expectations to test
    test['target matrix']['test'] = target_t
    test['input matrix']['test'] = inputs_t
    test['output matrix']['test'] = activation_t[:, -No:]
    test['activation matrix']['test'] = activation_t
    test['error matrix']['test'] = error[samples_t]

    # generate sample versions
    Mf = PackedMatrix(
        np.vstack((pk.packmat(inputs),
                   pk.packmat(target))),
        Ne=inputs.shape[0], Ni=Ni)

    Ms, Mt = partition_packed(Mf, samples)

    # add states to test
    test['state'] = {
        'full': BNState(gates, Mf),
        'sample': BNState(gates, Ms),
        'test': BNState(gates, Mt)
    }

    return test


@fixture(params=['sample', 'full'])
def sample_type(request):
    return request.param


@fixture(params=TEST_NETWORKS)
def state_harness(request):
    return harness_to_fixture(request.param)


@fixture(params=TEST_NETWORKS)
def state_params(request):
    fname = request.param
    if fname not in HARNESS_CACHE:
        with open(fname) as stream:
            harness = yaml.safe_load(stream)
            HARNESS_CACHE[fname] = harness
    # copy to ensure harness cache remains correct
    test = deepcopy(HARNESS_CACHE[fname])

    Ni = test['Ni']
    if test['target function'] in ['zero', 'unary_and', 'unary_or']:
        test['Nb'] = Ni
    else:
        test['Nb'] = Ni // 2
    test['operator'] = operator_map[test['target function']]

    indices_s = np.array(test['samples'], dtype=np.uint32)
    indices_f = np.arange(2**Ni, dtype=np.uint32)

    test['exclude'] = {'full': False, 'sample': False, 'test': True}

    # add generated params to test
    test['indices'] = {
        'sample': indices_s,
        'full': indices_f,
        'test': indices_s
    }

    Ne_f = 2**Ni
    Ne_s = indices_s.size
    Ne_t = Ne_f - Ne_s

    test['window_size'] = {
        'sample': np.random.randint(1, max(2, Ne_s // PACKED_SIZE)),
        'full': np.random.randint(1, max(2, Ne_f // PACKED_SIZE)),
        'test': np.random.randint(1, max(2, Ne_t // PACKED_SIZE))
    }

    return test


@fixture(params=list(harnesses_with_property('invariant under single move')))
def single_move_invariant(request):
    return harness_to_fixture(request.param)


@fixture(params=list(harnesses_with_property('invariant under multiple moves')))
def multiple_move_invariant(request):
    return harness_to_fixture(request.param)


MoveAndExpected = namedtuple('MoveAnExpected', ['move', 'expected'])


# ##################### TEST HELPERS ##################### #
def build_instance(instance_dict, sample_type, field):
    state = instance_dict['state'][sample_type]
    expected = np.array(instance_dict[field][sample_type], dtype=np.uint8)
    return state, expected, lambda mat: pk.unpackmat(mat, state.Ne)


def output_different(instance):
    state, _, eval_func = build_instance(instance, 'full', 'error matrix')
    for k in range(10):
        old_error = eval_func(state.error_matrix)
        state.move_to_random_neighbour()
        assert not np.array_equal(eval_func(state.error_matrix), old_error)
        state.revert_move()


def run_instance(instance, state):
    func_id = function_from_name(instance['function'])
    expected = instance['value']
    name = state.add_function(func_id)
    actual = state.function_value(name)
    np.testing.assert_array_almost_equal(expected, actual)


# ##################### FIXTURES ##################### #
@fixture(params=TEST_NETWORKS)
def state(request):
    return harness_to_fixture(request.param)


@fixture
def single_layer_zero():
    instance = harness_to_fixture(
        'boolnet/test/networks/single_layer_zero.yaml')

    instance = copy(instance)

    test_case = instance['multiple_moves_test_case']

    updated_test_case = []
    for step in test_case:
        move = step['move']
        expected = np.array(step['expected'], dtype=np.uint8)
        updated_test_case.append(
            MoveAndExpected(move=move, expected=expected))

    instance['multiple_moves_test_case'] = updated_test_case
    return instance


# ################### Exception Testing ################### #
@mark.parametrize("Ni, Tshape, Ne", [
    (1, (2, 4), 4),     # net.No != #tgts
    (2, (1, 4), 4),     # net.Ni != #inps
    (2, (2, 4), 4)      # both
])
def test_construction_exceptions(Ni, Tshape, Ne):
    M = PackedMatrix(np.vstack((
        all_possible_inputs(Ni),
        packed_zeros(Tshape))), Ne, Ni)
    with raises(ValueError):
        BNState([(0, 1, 3)], M)


# ################### Functionality Testing ################### #


def test_input_matrix(state, sample_type):
    state, expected, eval_func = build_instance(state, sample_type,
                                                'input matrix')
    actual = eval_func(state.input_matrix)
    np.testing.assert_array_equal(expected, actual)


def test_target_matrix(state, sample_type):
    state, expected, eval_func = build_instance(state, sample_type,
                                                'target matrix')
    actual = eval_func(state.target_matrix)
    np.testing.assert_array_equal(expected, actual)


def test_output_matrix(state, sample_type):
    state, expected, eval_func = build_instance(state, sample_type,
                                                'output matrix')
    actual = eval_func(state.output_matrix)
    np.testing.assert_array_equal(expected, actual)


def test_activation_matrix(state, sample_type):
    state, expected, eval_func = build_instance(state, sample_type,
                                                'activation matrix')
    actual = eval_func(state.activation_matrix)
    np.testing.assert_array_equal(expected, actual)


def test_error_matrix(state, sample_type):
    state, expected, eval_func = build_instance(state, sample_type,
                                                'error matrix')
    actual = eval_func(state.error_matrix)
    np.testing.assert_array_equal(expected, actual)


def test_single_move_output_different(single_move_invariant):
    output_different(single_move_invariant)


def test_multiple_move_output_different(multiple_move_invariant):
    output_different(multiple_move_invariant)


def test_move_with_initial_evaluation(single_layer_zero):
    state, expected, eval_func = build_instance(
        single_layer_zero, 'full', 'error matrix')

    actual = eval_func(state.error_matrix)
    np.testing.assert_array_equal(expected, actual)

    test_case = single_layer_zero['multiple_moves_test_case'][4]
    expected = test_case.expected

    state.apply_move(test_case.move)
    actual = eval_func(state.error_matrix)
    np.testing.assert_array_equal(expected, actual)


def test_multiple_moves_error_matrix(single_layer_zero):
    state, _, eval_func = build_instance(
        single_layer_zero, 'full', 'error matrix')

    test_case = single_layer_zero['multiple_moves_test_case']

    for move, expected in test_case:
        state.apply_move(move)
        actual = eval_func(state.error_matrix)
        np.testing.assert_array_equal(expected, actual)


def test_multiple_reverts_error_matrix(single_layer_zero):
    state, _, eval_func = build_instance(
        single_layer_zero, 'full', 'error matrix')

    test_case = single_layer_zero['multiple_moves_test_case']

    for move, _ in test_case:
        state.apply_move(move)

    for _, expected in reversed(test_case):
        actual = eval_func(state.error_matrix)
        np.testing.assert_array_equal(expected, actual)
        state.revert_move()


def test_pre_evaluated_network(state):
    state_s, expected, eval_func_s = build_instance(
        state, 'sample', 'activation matrix')

    for i in range(10):
        state_s.move_to_random_neighbour()
        state_s.evaluate()
    state_s.revert_all_moves()

    # check sample state is still giving original results
    actual = eval_func_s(state_s.activation_matrix)
    np.testing.assert_array_equal(expected, actual)

    state_f, expected, eval_func_f = build_instance(
        state, 'full', 'activation matrix')

    # check full state is still giving original results
    state_f.set_gates(state_s.gates)
    actual = eval_func_f(state_f.activation_matrix)

    np.testing.assert_array_equal(expected, actual)


def test_from_operator_combined_attributes(state_params, sample_type):
    state = state_from_operator(
        gates=state_params['gates'],
        indices=state_params['indices'][sample_type],
        Nb=state_params['Nb'],
        No=state_params['No'],
        operator=state_params['operator'],
        exclude=state_params['exclude'][sample_type]
    )

    assert state.Ni == state_params['Ni']
    assert state.No == state_params['No']
    assert state.Ng == len(state_params['gates'])
    assert state.zero_mask == pk.generate_end_mask(state.Ne)


def test_from_operator_func_value(state_params, sample_type):
    for instance in state_params['instances'][sample_type]:
        state = state_from_operator(
            gates=state_params['gates'],
            indices=state_params['indices'][sample_type],
            Nb=state_params['Nb'], No=state_params['No'],
            operator=state_params['operator'],
            exclude=state_params['exclude'][sample_type])
        run_instance(instance, state)


def test_function_value(state_harness, sample_type):
    state = state_harness['state'][sample_type]
    for instance in state_harness['instances'][sample_type]:
        run_instance(instance, state)
