import numpy as np
import bitpacking.packing as pk


class PackedMatrix(np.ndarray):
    def __new__(cls, input_array, Ne, Ni=0):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.Ne = Ne
        obj.Ni = Ni
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.Ne = getattr(obj, 'Ne', None)
        self.Ni = getattr(obj, 'Ni', None)

    @property
    def No(self):
        return self.shape[0] - self.Ni

    def split(self):
        return np.split(self, [self.Ni])

    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super(PackedMatrix, self).__reduce__()
        # Create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.Ne, self.Ni)
        # Return a tuple that replaces the parent's __setstate__ with our own
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        self.Ne = state[-2]  # Set the info attribute
        self.Ni = state[-1]  # Set the info attribute
        # Call the parent's __setstate__ with the other tuple elements.
        super(PackedMatrix, self).__setstate__(state[0:-2])


def partition_packed(matrix, indices):
    M_trg, M_test = pk.partition_columns(matrix, matrix.Ne, indices)

    # return the PackedMatrix types to provide meta-data
    Ne = indices.shape[0]
    training_matrix = PackedMatrix(M_trg, Ne=Ne, Ni=matrix.Ni)
    test_matrix = PackedMatrix(M_test, Ne=matrix.Ne-Ne, Ni=matrix.Ni)

    return training_matrix, test_matrix


def sample_packed(matrix, indices, invert=False):
    sample = pk.sample_columns(matrix, matrix.Ne, indices, invert)
    Ne = matrix.Ne - indices.size if invert else indices.size
    return PackedMatrix(sample, Ne=Ne, Ni=matrix.Ni)


def unpack(packed_matrix, transpose=True):
    return pk.unpackmat(packed_matrix, packed_matrix.Ne, transpose=transpose)


def inverse_permutation(permutation):
    inverse = np.zeros_like(permutation)
    for i, p in enumerate(permutation):
        inverse[p] = i
    return inverse


def rank_with_ties_broken(ranking_with_ties):
    new_ranking = np.zeros_like(ranking_with_ties)
    ranks, counts = np.unique(ranking_with_ties, return_counts=True)
    for rank, count in zip(ranks, counts):
        indices = np.where(ranking_with_ties == rank)
        perm = np.random.permutation(count) + rank
        new_ranking[indices] = perm
    return new_ranking


def order_from_rank(ranking_with_ties):
    ''' Converts a ranking with ties into an ordering,
        breaking ties with uniform probability.'''
    ranking_without_ties = rank_with_ties_broken(ranking_with_ties)
    # orders and rankings are inverse when no ties are present
    return inverse_permutation(ranking_without_ties)
