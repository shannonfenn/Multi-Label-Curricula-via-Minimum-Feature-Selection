# cython: language_level=3
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

cpdef enum Function:
    E1
    E2,
    E3,
    E4,
    E5,
    E6,
    E7,
    E1_MCC,
    E2_MCC,
    E6_MCC,
    E6_THRESHOLDED,
    E3_GENERAL,
    E6_GENERAL,
    CORRECTNESS,
    PER_OUTPUT_ERROR,
    PER_OUTPUT_MCC
