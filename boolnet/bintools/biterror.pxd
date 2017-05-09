# cython: language_level=3
import cython
cimport numpy as np
from bitpacking.packing cimport packed_type_t

cdef confusion(packed_type_t[:] errors, packed_type_t[:] target, size_t Ne,
               packed_type_t[:] TP_buf, packed_type_t[:] FP_buf, packed_type_t[:] FN_buf)

cdef double matthews_corr_coef(size_t TP, size_t TN, size_t FP, size_t FN)


cdef class Evaluator:
    cdef size_t Ne, No, cols
    cdef double divisor


cdef class PerOutputMCC(Evaluator):
    cdef packed_type_t[:] tp_buffer, fp_buffer, fn_buffer
    cdef np.uint8_t[:] errors_exist
    cdef double[:] mcc
    cdef packed_type_t end_mask

    cpdef double[:] evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T)


cdef class PerOutputMean(Evaluator):
    cdef double[:] accumulator

    cpdef double[:] evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T)


cdef class MeanMCC(Evaluator):
    cdef PerOutputMCC per_output_evaluator

    cpdef double evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T)


cdef class Correctness(Evaluator):
    cdef packed_type_t[:] row_disjunction

    cpdef double evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T)


cdef class E1(Evaluator):
    cpdef double evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T)


cdef class E2(Evaluator):
    cdef readonly double[:] weight_vector

    cpdef double evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T)


cdef class E2MCC(E2):
    cdef PerOutputMCC per_output_evaluator

    cpdef double evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T)


cdef class E3(Evaluator):
    cdef packed_type_t[:] row_disjunction

    cpdef double evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T)


cdef class E3General(Evaluator):
    cdef list dependencies
    cdef packed_type_t[:, :] disjunctions

    cpdef double evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T)


cdef class E4(Evaluator):
    cdef size_t end_subtractor

    cpdef double evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T)


cdef class E5(Evaluator):
    cdef size_t row_width, end_subtractor

    cpdef double evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T)


cdef class E6(Evaluator):
    cpdef double evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T)


cdef class E6MCC(Evaluator):
    cdef PerOutputMCC per_output_evaluator

    cpdef double evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T)


cdef class E6Thresholded(Evaluator):
    cpdef double threshold
    cdef PerOutputMean per_output_evaluator

    cpdef double evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T)


cdef class E6General(Evaluator):
    cpdef double threshold
    cdef list dependencies
    cdef PerOutputMean per_output_evaluator

    cpdef double evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T)


cdef class E7(Evaluator):
    cpdef double evaluate(self, packed_type_t[:, ::1] E, packed_type_t[:, ::1] T)
