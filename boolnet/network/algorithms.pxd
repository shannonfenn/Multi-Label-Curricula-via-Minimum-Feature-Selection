cimport numpy as np

cdef size_t sample_bool(np.uint8_t[:] M, size_t end=*)

#cdef size_t sample_masked_bool(np.uint8_t[:] M, np.uint8_t[:] mask, size_t end=*)

cpdef connected_sources(np.uint32_t[:, :] gates, np.uint8_t[:] connected,
                        size_t Ni, size_t No)