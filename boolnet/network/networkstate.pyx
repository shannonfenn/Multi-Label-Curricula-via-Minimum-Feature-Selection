# cython: language_level=3
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False
# distutils: language = c++
import cython
import numpy as np
cimport numpy as np
from copy import copy
from math import ceil

from bitpacking.packing import packed_type
from bitpacking.packing cimport (
    packed_type_t, generate_end_mask, f_type, function_list, PACKED_SIZE)

from boolnet.network.boolnet cimport BoolNet, Move
from boolnet.bintools.functions import function_name
from boolnet.bintools.functions cimport Function
from boolnet.bintools.biterror import EVALUATORS
from boolnet.bintools.operator_iterator cimport OpExampleIterFactory
from boolnet.bintools.example_generator cimport (
    PackedExampleGenerator, packed_from_operator)


cpdef state_from_operator(gates, indices, Nb, No, operator, order=None, exclude=False):
    M = packed_from_operator(indices, Nb, No, operator, exclude)

    if order is None:
        order = np.arange(No, dtype=np.uintp)

    M[-No:, :] = M[-No:, :][order, :]

    return BNState(gates, M)


cdef class BNState:
    cdef:
        readonly BoolNet network
        readonly size_t Ne, cols
        packed_type_t[:, :] activation, inputs, outputs, target, error
        readonly packed_type_t zero_mask
        dict err_evaluators
        readonly size_t invalid_start
        readonly bint evaluated

    def __init__(self, gates, problem_matrix):
        ''' Sets up the activation and error matrices for a new network.
            Note: This copies the provided network, so do not expect modifications
                  to pass through transparently without reacquiring the new alias.'''
        Ni = problem_matrix.Ni
        No = problem_matrix.shape[0] - Ni
        Ne = problem_matrix.Ne
        cols = problem_matrix.shape[1]

        self.network = BoolNet(gates, Ni, No)
        Ng = self.network.Ng

        self.Ne = Ne
        self.cols = cols

        self.zero_mask = generate_end_mask(Ne)

        # instantiate a matrix for activation
        self.activation = np.zeros((Ng + Ni, cols), dtype=packed_type)

        # create input and output view into activation matrix
        self.inputs = self.activation[:Ni, :]
        if No != 0:
            self.outputs = self.activation[-<int>No:, :]
        else:
            self.outputs = np.empty((No, cols), dtype=packed_type)

        self.target = np.zeros((No, cols), dtype=packed_type)
        # instantiate matrices for error
        self.error = np.empty_like(self.target)

        self.err_evaluators = dict()

        self.evaluated = False
        self.invalid_start = 0
        self._check_state_invariants()

        # buffer view for copying
        cdef packed_type_t[:, :] P = problem_matrix

        self.inputs[...] = P[:Ni, :]
        self.target[...] = P[Ni:, :]

        # just in case
        self._apply_zero_mask(self.activation)   # masks input/output too (they're views)
        self._apply_zero_mask(self.target)

    def __copy__(self):
        # we don't want copying
        raise NotImplementedError

    property Ni:
        def __get__(self):
            return self.network.Ni

    property No:
        def __get__(self):
            return self.network.No

    property Ng:
        def __get__(self):
            return self.network.Ng

    property gates:
        def __get__(self):
            return self.network.gates

    property representation:
        def __get__(self):
            return self.network

    property guiding_functions:
        def __get__(self):
            # conversion to list prevents silly copy bugs
            return list(self.err_evaluators.keys())

    property input_matrix:
        def __get__(self):
            return self.inputs

    property target_matrix:
        def __get__(self):
            return self.target

    property activation_matrix:
        def __get__(self):
            self.evaluate()
            return self.activation

    property output_matrix:
        def __get__(self):
            self.evaluate()
            return self.outputs

    property error_matrix:
        def __get__(self):
            self.evaluate()
            return self.error

    cpdef connected_gates(self):
        return self.network.connected_gates()

    cpdef connected_sources(self):
        return self.network.connected_sources()

    cpdef add_function(self, Function function, name='', params={}):
        eval_class = EVALUATORS[function]
        if not name:
            name = function_name(function)
        self.err_evaluators[name] = eval_class(self.Ne, self.No, **params)
        self.evaluated = False
        return name

    cpdef function_value(self, name):
        if name not in self.err_evaluators:
            raise ValueError('No evaluator with name: {}'.format(name))
        if not self.evaluated:
            self.evaluate()
        return self.err_evaluators[name].evaluate(self.error, self.target)

    cpdef set_gates(self, np.uint32_t[:, :] gates):
        # force reevaluation
        self.evaluated = False
        self.invalid_start = 0
        self.network.set_gates(gates)
        self._check_state_invariants()

    cpdef force_reevaluation(self):
        self.evaluated = False
        self.invalid_start = 0
    
    cpdef randomise(self):
        self.network.randomise()
        # indicate the network must be reevaluated
        self.evaluated = False
        self.invalid_start = 0

    cpdef move_to_random_neighbour(self):
        self.apply_move(self.network.random_move())

    cpdef apply_move(self, Move move):
        # indicate the network must be reevaluated
        self.evaluated = False
        if self.evaluated:
            self.invalid_start = move.gate
        else:
            self.invalid_start = min(self.invalid_start,
                                              move.gate)
        self.network.apply_move(move)

    cpdef revert_move(self):
        cdef Move inverse_move
        inverse_move = self.network.inverse_moves.back()
        self.network.revert_move()
        if self.evaluated:
            self.invalid_start = inverse_move.gate
        else:
            # if multiple moves are undone the network needs to
            # be recomputed from the earliest gate ever changed
            self.invalid_start = min(
                self.invalid_start, inverse_move.gate)
        self.evaluated = False

    cpdef revert_all_moves(self):
        self.network.revert_all_moves()
        self.evaluated = False
        self.invalid_start = 0

    cpdef clear_history(self):
        self.network.clear_history()

    ############################### Evaluation methods ###############################
    cpdef evaluate(self):
        ''' Evaluate the activation and error matrices for the network
            getting node TFs from network. '''
        cdef:
            size_t Ni, No, Ng, cols, c, g, o, src1, src2, start
            packed_type_t[:, :] activation, outputs, error, target
            np.uint32_t[:, :] gates
            f_type func

        if self.evaluated:
            return

        Ng = self.network.Ng
        Ni = self.network.Ni
        No = self.network.No
        # local memoryviews to avoid 'self' evaluation later
        activation = self.activation
        outputs = self.outputs
        error = self.error
        target = self.target
        gates = self.network.gates
        
        cols = activation.shape[1]

        # evaluate the state matrix
        start = self.invalid_start

        for g in range(start, Ng):
            src1 = gates[g, 0]
            src2 = gates[g, 1]
            func = function_list[gates[g, 2]]
            for c in range(cols):
                activation[Ni+g, c] = func(activation[src1, c], activation[src2, c])

        # evaluate the error matrix
        for o in range(No):
            for c in range(cols):
                error[o, c] = (target[o, c] ^ outputs[o, c])

        # masking the activation does the output as well (since it is a view)
        self._apply_zero_mask(self.activation)
        self._apply_zero_mask(self.error)

        self.evaluated = True

    cdef void _apply_zero_mask(self, packed_type_t[:,:] matrix):
        # when evaluating make a zeroing-mask '11110000' to AND the last
        # column in the error matrix with to clear the value back to zero
        cdef size_t r, rows, cols

        rows, cols = matrix.shape[0], matrix.shape[1]

        for r in range(rows):
            matrix[r, cols-1] &= self.zero_mask

    cdef _check_state_invariants(self):
        cdef size_t Ng, Ni, No
        Ng = self.network.Ng
        Ni = self.network.Ni
        No = self.network.No
        if self.cols == 0 or self.Ne == 0:
            raise ValueError('Zero value for cols ({}) or Ne ({}).'.
                             format(self.cols, self.Ne))
        if self.activation.shape[0] != Ni + Ng:
            raise ValueError('Activation length ({}) does not match Ni+Ng ({}).'.
                             format(self.activation.shape[0], Ni + Ng))
        if self.inputs.shape[0] != Ni:
            raise ValueError('Inputs length ({}) does not match Ni ({}).'.
                             format(self.inputs.shape[0], Ni))
        if self.outputs.shape[0] != No:
            raise ValueError('Outputs length ({}) does not match No ({}).'.
                             format(self.outputs.shape[0], No))
        if self.target.shape[0] != No:
            raise ValueError('Target length ({}) does not match No ({}).'.
                             format(self.target.shape[0], No))
        if self.error.shape[0] != No:
            raise ValueError('Error length ({}) does not match No ({}).'.
                             format(self.error.shape[0], self.No))
        if self.activation.shape[1] != self.cols:
            raise ValueError('Activation column width ({}) != cols ({}).'.
                             format(self.activation.shape[1], self.cols))
        if self.inputs.shape[1] != self.cols:
            raise ValueError('Inputs column width ({}) != cols ({}).'.
                             format(self.inputs.shape[1], self.cols))
        if self.outputs.shape[1] != self.cols:
            raise ValueError('Outputs column width ({}) != cols ({}).'.
                             format(self.outputs.shape[1], self.cols))
        if self.target.shape[1] != self.cols:
            raise ValueError('Target column width ({}) != cols ({}).'.
                             format(self.target.shape[1], self.cols))
        if self.error.shape[1] != self.cols:
            raise ValueError('Error column width ({}) != cols ({}).'.
                             format(self.error.shape[1], self.cols))

