# cython: language_level=3
# distutils: language = c++
import numpy as np
cimport numpy as np
from libcpp.deque cimport deque
from boolnet.exptools.fastrand cimport random_uniform_int
cimport boolnet.network.algorithms as algorithms


cdef class BoolNet:    
    def __init__(self, gates, size_t Ni, size_t No):
        # copy gate array
        self.gates = np.array(gates, dtype=np.uint32, copy=True)

        # Store size values
        self.Ng = self.gates.shape[0]
        self.Ni = Ni
        self.No = No

        self.connected = np.zeros(self.Ng + Ni, dtype=np.uint8)

        self._check_network_invariants()
        self._update_connected()

    def __copy__(self):
        cdef BoolNet bn
        bn = BoolNet(self.gates, self.Ni, self.No)
        bn.connected[...] = self.connected
        bn.inverse_moves = self.inverse_moves
        return bn

    def __str__(self):
        return ('Ni: {} Ng: {} gates:\n{}\n').format(
            self.Ni, self.Ng, np.asarray(self.gates))

    cpdef max_node_depths(self):
        # default to stored value of No
        cdef size_t Ni, No, Ng, g
        No = self.No
        Ng = self.Ng
        Ni = self.Ni
        depths = [0] * (Ng + Ni)
        for g in range(Ng):
            # don't include the transfer function
            source_depths = [depths[inp] for inp in self.gates[g][:-1]]
            depths[g + Ni] = max(source_depths) + 1
        # return the last No values
        return depths[-No:]

    cpdef set_gates(self, np.uint32_t[:, :] gates):
        self.gates[...] = gates
        # check invariants hold
        self._check_network_invariants()

    cpdef randomise(self):
        # this has the side effect of clearing the history since we don't
        # want to compute all the inverse moves
        cdef size_t g, Ni, No, Ng
        Ni = self.Ni
        No = self.No
        Ng = self.Ng
        for g in range(self.Ng):
            self.gates[g, 0] = random_uniform_int(min(g, Ng - No) + Ni)
            self.gates[g, 1] = random_uniform_int(min(g, Ng - No) + Ni)
        self.clear_history()

    cpdef connected_gates(self):
        self._update_connected()
        return self.connected[self.Ni:]

    cpdef connected_sources(self):
        self._update_connected()
        return self.connected

    cpdef _update_connected(self):
        algorithms.connected_sources(self.gates, self.connected,
                                     self.Ni, self.No)

    cpdef move_to_random_neighbour(self):
        self.apply_move(self.random_move())

    cpdef Move random_move(self) except +:
        ''' Returns a 3-tuple of unsigned ints (gate, terminal, new_source. '''
        # In order to ensure a useful modification is generated we make
        # sure to only modify connected gates
        cdef:
            size_t gate, terminal, shift
            size_t cur_source, new_source, Ni, Ng, No
            np.uint8_t temp
            Move move
        Ni = self.Ni
        Ng = self.Ng
        No = self.No

        # pick a random gate to move
        gate = algorithms.sample_bool(self.connected_gates())

        # pick single random input connection to move
        terminal = random_uniform_int(2)
        # pick new source for connection randomly
        # can only come from earlier node to maintain feedforwardness
        cur_source = self.gates[gate][terminal]

        # decide how much to shift the input
        # (gate can only connect to previous gate or input)
        # (output gates cannot connect to each other)
        shift = random_uniform_int(min(gate, Ng - No) + Ni - 1) + 1
        # Get shifted connection
        new_source = (cur_source + shift) % (min(gate, Ng - No) + Ni)

        return Move(gate, terminal, new_source)

    cpdef apply_move(self, Move move):
        # expects iterable with the form (gate, terminal, new_source)
        # record the inverse move
        cdef Move inverse
        inverse.gate = move.gate
        inverse.terminal = move.terminal
        inverse.new_source = self.gates[move.gate][move.terminal]
        self.inverse_moves.push_back(inverse)

        # modify the connection
        self.gates[move.gate][move.terminal] = move.new_source

    cpdef revert_move(self):
        cdef Move inverse
        if not self.inverse_moves.empty():
            inverse = self.inverse_moves.back()
            self.inverse_moves.pop_back()
            self.gates[inverse.gate][inverse.terminal] = inverse.new_source
        else:
            raise RuntimeError('Tried to revert with empty inverse move list.')

    cpdef revert_all_moves(self):
        while not self.inverse_moves.empty():
            self.revert_move()

    cpdef clear_history(self):
        self.inverse_moves.clear()

    cpdef history_empty(self):
        return self.inverse_moves.empty()

    cdef _check_network_invariants(self):
        #if self.Ng == 0:
        #    raise ValueError('Empty initial gates list')
        if self.Ni == 0:
            raise ValueError('Invalid Ni ({})'.format(self.Ni))
        #if self.No <= 0:
        #    raise ValueError('Invalid No ({})'.format(self.No))
        if self.No > self.Ng:
            raise ValueError('No > Ng ({}, {})'.format(self.No, self.Ng))
        if self.gates.ndim != 2:
            raise ValueError('gates must be 2D')
        if self.gates.shape[0] != self.Ng or self.gates.shape[1] != 3:
            raise ValueError('Wrong shape ({}, {}) for gate matrix, Ng={}.'.
                format(self.gates.shape[0], self.gates.shape[1], self.Ng))
        if self.Ng > 0 and max(self.gates[:, 2]) > 15:
            raise ValueError('Invalid transfer functions: {}'.format(
                max(self.gates[:, 2])))
