from bitpacking import packing as pk
from boolnet.bintools.functions import function_from_name
import numpy as np
from boolnet.bintools.biterror import EVALUATORS


def test_function_value(error_matrix_harness):
    Ne = error_matrix_harness['Ne']
    E = error_matrix_harness['packed error matrix']
    T = np.zeros_like(E)
    No, _ = E.shape

    for test in error_matrix_harness['tests']:
        order = test['order']
        func_id = function_from_name(test['function'])
        expected = test['value']

        if order == 'lsb':
            order = np.arange(No, dtype=np.uintp)
        elif order == 'msb':
            order = np.arange(No, dtype=np.uintp)[::-1]

        E_ = E[order, :]
        T_ = T[order, :]

        eval_class = EVALUATORS[func_id]

        error_evaluator = eval_class(Ne, No)
        actual = error_evaluator.evaluate(E_, T_)
        np.testing.assert_array_almost_equal(actual, expected)


def test_e3_general():
    E = np.array([[1, 0, 1, 1, 0],
                  [0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0],
                  [0, 1, 0, 0, 0],
                  [0, 1, 0, 0, 0],
                  [0, 1, 0, 0, 1],
                  [0, 0, 0, 0, 1],
                  [0, 0, 1, 1, 0],
                  [0, 0, 1, 1, 0]], dtype=np.uint8)
    # E = np.array([[1, 0, 1, 1, 1],
    #               [0, 0, 0, 0, 0],
    #               [0, 1, 1, 1, 1],
    #               [0, 1, 1, 1, 1],
    #               [0, 1, 1, 1, 1],
    #               [0, 1, 1, 1, 1],
    #               [0, 0, 0, 0, 1],
    #               [0, 0, 1, 1, 1],
    #               [0, 0, 1, 1, 1]], dtype=np.uint8)

    Ne, No = E.shape
    Ep = pk.packmat(E)
    Tp = np.zeros_like(Ep)

    eval_class = EVALUATORS[function_from_name('e3_general')]

    dependencies = [[], [], [0, 1], [0, 1, 2], [0, 1, 2]]

    error_evaluator = eval_class(Ne, No, dependencies)
    actual = error_evaluator.evaluate(Ep, Tp)
    np.testing.assert_array_almost_equal(actual, 27/45)


def test_e6_general():
    E = np.array([[0, 0, 1, 1, 0, 0],
                  [0, 0, 0, 0, 0, 1],
                  [0, 1, 0, 0, 0, 1],
                  [0, 1, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 1],
                  [0, 1, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1, 1],
                  [0, 0, 1, 1, 0, 0],
                  [0, 0, 1, 1, 0, 1]], dtype=np.uint8)
    # E = np.array([[0, 0, 1, 1, 1, 0],
    #               [0, 0, 1, 1, 1, 1],
    #               [0, 1, 1, 1, 1, 1],
    #               [0, 1, 1, 1, 1, 0],
    #               [0, 1, 1, 1, 1, 1],
    #               [0, 1, 1, 1, 1, 0],
    #               [0, 0, 1, 1, 1, 1],
    #               [0, 0, 1, 1, 1, 0],
    #               [0, 0, 1, 1, 1, 1]], dtype=np.uint8)

    Ne, No = E.shape
    Ep = pk.packmat(E)
    Tp = np.zeros_like(Ep)

    eval_class = EVALUATORS[function_from_name('e6_general')]

    dependencies = [[], [], [0, 1], [0, 1, 2], [0, 1, 2], [0]]

    error_evaluator = eval_class(Ne, No, dependencies)
    actual = error_evaluator.evaluate(Ep, Tp)
    np.testing.assert_array_almost_equal(actual, 36/54)
# @pytest.mark.python
# def test_error(error_matrix_harness, function):
#     from BoolNet.BitError import function_value
#     actual = function_value(error_matrix_harness['error matrix'], function)
#     expected = error_matrix_harness[function_name(function)]
#     assert np.array_equal(actual, expected)


# @pytest.mark.gpu
# def test_error_gpu(error_matrix_harness, function):
#     import pycuda.driver as cuda
#     import pycuda.autoinit
#     from BoolNet.BitErrorGPU import function_value_gpu, IMPLEMENTED_METRICS

#     err_mat = error_matrix_harness['error matrix']
#     d_error_mat = cuda.mem_alloc(err_mat.nbytes)
#     d_scratch_mat = cuda.mem_alloc(err_mat.shape[0] * 4)
#     Ne, No = err_mat.shape
#     Ne, No = np.int32(Ne), np.int32(No)
#     cuda.memcpy_htod(d_error_mat, err_mat)

#     expected = error_matrix_harness[function_name(function)]

#     if function in IMPLEMENTED_METRICS:
#         actual = function_value_gpu(d_error_mat, d_scratch_mat, Ne, No, function)
#         assert np.array_equal(actual, expected)
#     else:
#         with pytest.raises(NotImplementedError):
#             function_value_gpu(d_error_mat, d_scratch_mat, Ne, No, function)
#     d_scratch_mat.free()
#     d_error_mat.free()
