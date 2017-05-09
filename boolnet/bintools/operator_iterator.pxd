# cython: language_level=3
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

cpdef enum Operator:
    ZERO,
    AND,
    OR,
    UNARY_AND,
    UNARY_OR,
    ADD,
    SUB,
    MUL,


cpdef int num_operands(Operator op)


cdef class OpExampleIterFactory:
    cdef:
        size_t Ne, Nb, Ni, max_elements
        size_t[:] indices
        bint exclude
        Operator op
    cdef __check_operator(self, Operator op)


cdef class OpIterator:
    cdef size_t divisor, remaining


cdef class OpIncIterator(OpIterator):
    cdef object include_iter


cdef class ZeroIncIterator(OpIncIterator):
    pass


cdef class UnaryANDIncIterator(OpIncIterator):
    cdef size_t all_ones


cdef class UnaryORIncIterator(OpIncIterator):
    pass


cdef class ANDIncIterator(OpIncIterator):
    pass


cdef class ORIncIterator(OpIncIterator):
    pass


cdef class AddIncIterator(OpIncIterator):
    pass


cdef class SubIncIterator(OpIncIterator):
    pass


cdef class MulIncIterator(OpIncIterator):
    pass


cdef class OpExcIterator(OpIterator):    
    cdef:
        size_t index, ex_index, total_elements
        object ex_iter

    cdef void _sync(self)


cdef class ZeroExcIterator(OpExcIterator):
    pass


cdef class UnaryANDExcIterator(OpExcIterator):
    cdef size_t all_ones


cdef class UnaryORExcIterator(OpExcIterator):
    pass


cdef class ANDExcIterator(OpExcIterator):
    pass


cdef class ORExcIterator(OpExcIterator):
    pass


cdef class AddExcIterator(OpExcIterator):
    pass


cdef class SubExcIterator(OpExcIterator):
    pass


cdef class MulExcIterator(OpExcIterator):
    pass
