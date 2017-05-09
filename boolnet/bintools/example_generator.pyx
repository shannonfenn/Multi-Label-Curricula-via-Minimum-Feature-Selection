# cython: language_level=3
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False
import numpy as np
from bitpacking.packing cimport packed_type_t, pack_chunk, PACKED_SIZE
from bitpacking.packing import packed_type
from boolnet.utils import PackedMatrix
from boolnet.bintools.operator_iterator cimport OpExampleIterFactory


cpdef packed_from_operator(indices, Nb, No, operator, exclude=False):
    cdef packed_type_t[:, :] inp, tgt

    ex_factory = OpExampleIterFactory(indices, Nb, operator, exclude)
    packed_factory = PackedExampleGenerator(ex_factory, No)

    Ni = packed_factory.Ni
    Ne = packed_factory.Ne

    chunks = Ne // PACKED_SIZE
    if Ne % PACKED_SIZE > 0:
        chunks += 1

    M = PackedMatrix(np.empty((Ni+No, chunks), dtype=packed_type), Ne, Ni)
    
    I, T = np.split(M, [Ni])
    packed_factory.reset()
    packed_factory.next_examples(I, T)
    return M


cdef class PackedExampleGenerator:
    def __init__(self, OpExampleIterFactory iterator_factory, size_t No):
        self.No = No
        self.Ne = iterator_factory.Ne
        self.Ni = iterator_factory.Ni
        self.iterator_factory = iterator_factory

        self.inp_block = np.zeros(PACKED_SIZE, dtype=packed_type)
        self.tgt_block = np.zeros(PACKED_SIZE, dtype=packed_type)
        self.reset()

    cpdef reset(self):
        self.example_iter = iter(self.iterator_factory)

    cpdef next_examples(self, packed_type_t[:, :] inputs, packed_type_t[:, :] target):
        cdef size_t i, remaining, remaining_blocks, blocks
        remaining = len(self.example_iter)
        
        if remaining == 0:
            raise IndexError('ExampleGenerator - past end of examples.')
        if inputs.shape[0] != self.Ni:
            raise ValueError('ExampleGenerator - inputs does not match Ni in shape.')
        if target.shape[0] != self.No:
            raise ValueError('ExampleGenerator - target does not match No in shape.')
        
        blocks = inputs.shape[1]
        remaining_blocks = remaining // PACKED_SIZE
        if remaining % PACKED_SIZE:
            remaining_blocks += 1

        for i in range(min(blocks, remaining_blocks)):
            self._get_block(inputs, target, i)

    def __bool__(self):
        return len(self.example_iter) > 0

    # cdef void _get_block(self, packed_type_t[:, :] inputs, packed_type_t[:, :] target, size_t col):
    cdef _get_block(self, packed_type_t[:, :] inputs, packed_type_t[:, :] target, size_t block):
        cdef size_t remaining

        remaining = min(len(self.example_iter), PACKED_SIZE)

        for i in range(remaining):
            self.inp_block[i], self.tgt_block[i] = next(self.example_iter)

        for i in range(remaining, PACKED_SIZE):
            self.inp_block[i], self.tgt_block[i] = 0, 0

        pack_chunk(self.inp_block, inputs, self.Ni, block)
        pack_chunk(self.tgt_block, target, self.No, block)
