# cython: language_level=3
# cython: profile=False
# distutils: libraries = gsl, gslcblas

from cython_gsl cimport gsl_rng, gsl_rng_set, gsl_rng_alloc, gsl_rng_mt19937, gsl_rng_uniform_int
import cython

cdef gsl_rng *R = gsl_rng_alloc(gsl_rng_mt19937)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def seed(unsigned long int s):
    gsl_rng_set(R, s)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef unsigned int random_uniform_int(unsigned int N):
    if N > 0:
        return gsl_rng_uniform_int(R, N)
    else:
        return 0