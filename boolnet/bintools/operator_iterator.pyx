# cython: language_level=3
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False
import numpy as np
cimport numpy as np


def operator_from_name(name):
    if name == 'zero':
        return ZERO
    elif name == 'and':
        return AND
    elif name == 'or':
        return OR
    elif name == 'unary_and':
        return UNARY_AND
    elif name == 'unary_or':
        return UNARY_OR
    elif name == 'add':
        return ADD
    elif name == 'sub':
        return SUB
    elif name == 'mul':
        return MUL
    raise ValueError('No operator named: ' + name)


cpdef int num_operands(Operator op):
    if op in [AND, OR, ADD, SUB, MUL]:
        return 2
    else:
        return 1


cdef class OpExampleIterFactory:
    def __init__(self, indices, size_t Nb, Operator operator, bint exclude):
        self.__check_operator(operator)
        self.indices = np.array(indices, dtype=np.uintp)
        self.op = operator
        self.exclude = exclude
        self.Nb = Nb
        self.Ni = num_operands(operator) * Nb
        if self.exclude:
            self.max_elements = 2 ** self.Ni
            if self.max_elements < len(self.indices):
                raise ValueError('Exclude list larger than max_elements.')
            self.Ne = self.max_elements - len(self.indices)
            # sort the indices so that the iterator does not return incorrect examples
            self.indices = np.sort(self.indices)
        else:
            self.Ne = len(self.indices)

    def __iter__(self):
        if self.exclude:
            if self.op == ZERO:
                return ZeroExcIterator(self.indices, self.Nb, self.max_elements)
            elif self.op == UNARY_AND:
                return UnaryANDExcIterator(self.indices, self.Nb, self.max_elements)
            elif self.op == UNARY_OR:
                return UnaryORExcIterator(self.indices, self.Nb, self.max_elements)
            elif self.op == AND:
                return ANDExcIterator(self.indices, self.Nb, self.max_elements)
            elif self.op == OR:
                return ORExcIterator(self.indices, self.Nb, self.max_elements)
            elif self.op == ADD:
                return AddExcIterator(self.indices, self.Nb, self.max_elements)
            elif self.op == SUB:
                return SubExcIterator(self.indices, self.Nb, self.max_elements)
            elif self.op == MUL:
                return MulExcIterator(self.indices, self.Nb, self.max_elements)
        else:
            if self.op == ZERO:
                return ZeroIncIterator(self.indices, self.Nb)
            elif self.op == UNARY_AND:
                return UnaryANDIncIterator(self.indices, self.Nb)
            elif self.op == UNARY_OR:
                return UnaryORIncIterator(self.indices, self.Nb)
            elif self.op == AND:
                return ANDIncIterator(self.indices, self.Nb)
            elif self.op == OR:
                return ORIncIterator(self.indices, self.Nb)
            elif self.op == ADD:
                return AddIncIterator(self.indices, self.Nb)
            elif self.op == SUB:
                return SubIncIterator(self.indices, self.Nb)
            elif self.op == MUL:
                return MulIncIterator(self.indices, self.Nb)
            
    def __len__(self):
        return self.Ne

    cdef __check_operator(self, Operator op):
        if op not in [ZERO, AND, OR, UNARY_AND, UNARY_OR, ADD, SUB, MUL]:
            raise ValueError('Invalid operator value ({})'.format(op))



cdef class OpIterator:
    def __init__(self, size_t Nb, size_t Ne):
        if Nb == 0:
            raise ValueError('Zero bitwidth not allowed.')
        self.divisor = 2**Nb
        self.remaining = Ne

    def __iter__(self):
        return self

    def __len__(self):
        return self.remaining


cdef class OpIncIterator(OpIterator):
    def __init__(self, size_t[:] include_list, size_t Nb):
        super().__init__(Nb, include_list.size)
        self.include_iter = iter(include_list)


cdef class ZeroIncIterator(OpIncIterator):
    def __next__(self):
        cdef size_t inp, out 
        inp = next(self.include_iter)
        out = 0
        self.remaining -= 1
        return inp, out


cdef class UnaryANDIncIterator(OpIncIterator):
    def __init__(self, size_t[:] include_list, size_t Nb):
        super().__init__(include_list, Nb)
        self.all_ones = 2**Nb - 1

    def __next__(self):
        cdef size_t inp, out 
        inp = next(self.include_iter)
        out = 1 if inp == self.all_ones else 0
        self.remaining -= 1
        return inp, out


cdef class UnaryORIncIterator(OpIncIterator):
    def __next__(self):
        cdef size_t inp, out 
        inp = next(self.include_iter)
        out = 0 if inp == 0 else 1
        self.remaining -= 1
        return inp, out


cdef class ANDIncIterator(OpIncIterator):
    def __next__(self):
        cdef size_t inp, out 
        inp = next(self.include_iter)
        out = (inp // self.divisor) & (inp % self.divisor)
        self.remaining -= 1
        return inp, out


cdef class ORIncIterator(OpIncIterator):
    def __next__(self):
        cdef size_t inp, out 
        inp = next(self.include_iter)
        out = (inp // self.divisor) | (inp % self.divisor)
        self.remaining -= 1
        return inp, out


cdef class AddIncIterator(OpIncIterator):
    def __next__(self):
        cdef size_t inp, out 
        inp = next(self.include_iter)
        out = (inp // self.divisor) + (inp % self.divisor)
        self.remaining -= 1
        return inp, out


cdef class SubIncIterator(OpIncIterator):
    def __next__(self):
        cdef size_t inp, out, upper, lower
        inp = next(self.include_iter)
        upper = (inp // self.divisor)
        lower = (inp % self.divisor)
        if upper >= lower:
            out = upper - lower
        else:
            out = self.divisor - lower + upper
        self.remaining -= 1
        return inp, out


cdef class MulIncIterator(OpIncIterator):
    def __next__(self):
        cdef size_t inp, out 
        inp = next(self.include_iter)
        out = (inp // self.divisor) * (inp % self.divisor)
        self.remaining -= 1
        return inp, out


cdef class OpExcIterator(OpIterator):
    def __init__(self, size_t[:] exclude_list, size_t Nb, size_t total_elements):
        if total_elements < exclude_list.size:
            raise ValueError('Exclude list larger than total_elements.')
        super().__init__(Nb, total_elements - exclude_list.size)
        self.total_elements = total_elements
        self.ex_iter = iter(exclude_list)
        self.index = 0
        try:
            self.ex_index = next(self.ex_iter)
            self._sync()
        except StopIteration:
            self.ex_index = self.total_elements

    cdef void _sync(self):
        if self.ex_index < self.total_elements:
            try:
                while self.index == self.ex_index:
                    self.index += 1
                    self.ex_index = next(self.ex_iter)
            except StopIteration:
                self.ex_index = self.total_elements


cdef class ZeroExcIterator(OpExcIterator):
    def __next__(self):
        if self.remaining <= 0:
            raise StopIteration
        cdef size_t inp = self.index
        self.index += 1
        self.remaining -= 1
        self._sync()
        return inp, 0


cdef class UnaryANDExcIterator(OpExcIterator):
    def __init__(self, size_t[:] exclude_list, size_t Nb, size_t total_elements):
        super().__init__(exclude_list, Nb, total_elements)
        self.all_ones = 2**Nb - 1

    def __next__(self):
        if self.remaining <= 0:
            raise StopIteration
        cdef size_t inp, out 
        inp = self.index
        out = 1 if inp == self.all_ones else 0
        self.index += 1
        self.remaining -= 1
        self._sync()
        return inp, out


cdef class UnaryORExcIterator(OpExcIterator):
    def __next__(self):
        if self.remaining <= 0:
            raise StopIteration
        cdef size_t inp, out 
        inp = self.index
        out = 0 if inp == 0 else 1
        self.index += 1
        self.remaining -= 1
        self._sync()
        return inp, out


cdef class ANDExcIterator(OpExcIterator):
    def __next__(self):
        if self.remaining <= 0:
            raise StopIteration
        cdef size_t inp, out 
        inp = self.index
        out = (inp // self.divisor) & (inp % self.divisor)
        self.index += 1
        self.remaining -= 1
        self._sync()
        return inp, out


cdef class ORExcIterator(OpExcIterator):
    def __next__(self):
        if self.remaining <= 0:
            raise StopIteration
        cdef size_t inp, out 
        inp = self.index
        out = (inp // self.divisor) | (inp % self.divisor)
        self.index += 1
        self.remaining -= 1
        self._sync()
        return inp, out


cdef class AddExcIterator(OpExcIterator):
    def __next__(self):
        if self.remaining <= 0:
            raise StopIteration
        cdef size_t inp, out 
        inp = self.index
        out = (inp // self.divisor) + (inp % self.divisor)
        self.index += 1
        self.remaining -= 1
        self._sync()
        return inp, out


cdef class SubExcIterator(OpExcIterator):
    def __next__(self):
        if self.remaining <= 0:
            raise StopIteration
        cdef size_t inp, out, upper, lower
        inp = self.index
        upper = (inp // self.divisor)
        lower = (inp % self.divisor)
        if upper >= lower:
            out = upper - lower
        else:
            out = self.divisor - lower + upper
        self.index += 1
        self.remaining -= 1
        self._sync()
        return inp, out


cdef class MulExcIterator(OpExcIterator):
    def __next__(self):
        if self.remaining <= 0:
            raise StopIteration
        cdef size_t inp, out 
        inp = self.index
        out = (inp // self.divisor) * (inp % self.divisor)
        self.index += 1
        self.remaining -= 1
        self._sync()
        return inp, out