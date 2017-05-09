import yaml
import glob
import networkx as nx
import numpy as np
from numpy.testing import assert_array_equal, assert_array_less
from numpy.random import randint
from copy import copy
from itertools import chain
from pytest import fixture, raises
from pytest import mark
from boolnet.network.boolnet import BoolNet


# ############# Global helpers ################# #
def harness_to_fixture(stream):
    harness = yaml.safe_load(stream)
    Ni = harness['Ni']
    No = harness['No']
    gates = np.array(harness['gates'], np.uint32)
    # add network to harness
    harness['net'] = BoolNet(gates, Ni, No)
    return harness


# #################### Global fixtures #################### #
@fixture(scope='module')
def adder2():
    with open('boolnet/test/networks/adder2.yaml') as f:
        net = harness_to_fixture(f)
    return net['net']


@fixture(
    scope='module',
    params=glob.glob('boolnet/test/networks/*.yaml'))
def network_file_instance(request):
    return request.param


# #################### Exception Testing #################### #
class TestExceptions:
    # ############## Fixtures ##################### #

    @fixture
    def network(self):
        return BoolNet([(0, 1, 0)], 2, 1)

    @fixture(params=[
        ([], 0, 0),             # Ni = 0 with empty gates
        ([], 1, 1),             # No > 0 with empty gates
        ([[0, 0, 16]], 1, 1),         # tf > 15
        ([[0, 0, 1]], 0, 1),       # Ni invalid
        ([[0, 0, 1]], 1, 2),       # No > Ng
        ([0, 0, 1], 1, 1),         # gates not 2D
        ([[]], 0, 1),           # gates not mx3
        ([[1]], 0, 1),           # gates not mx3
        ([[0, 3]], 0, 1),          # gates not mx3
        ([[0, 0, 0, 0]], 0, 1)    # gates not mx3
        ])
    def invalid_construction(self, request):
        return request.param

    # ############## Tests ##################### #
    def test_construction_exceptions(self, invalid_construction):
        with raises(ValueError):
            BoolNet(*invalid_construction)

    def test_initial_revert_move_exception(self, network):
        with raises(RuntimeError):
            network.revert_move()

    def test_1_move_2_revert_exception(self, network):
        network.move_to_random_neighbour()
        network.revert_move()
        with raises(RuntimeError):
            network.revert_move()

    def test_n_move_n_plus_2_revert_exception(self, network):
        # n moves, n + 2 reverts
        for i in range(10):
            network.move_to_random_neighbour()
        for i in range(10):
            network.revert_move()
        with raises(RuntimeError):
            network.revert_move()
        with raises(RuntimeError):
            network.revert_move()

    def test_revert_after_revert_all_exception(self, network):
        # revert after revert all
        for i in range(10):
            network.move_to_random_neighbour()
        network.revert_all_moves()
        with raises(RuntimeError):
            network.revert_move()


# #################### Functional Testing #################### #
class TestFunctionality:
    # ############## Helper Methods ##################### #
    def graph_from_gates(self, gates, Ni, No):
        DG = nx.DiGraph()
        DG.add_nodes_from(range(Ni))
        edges = []
        for g in range(len(gates)):
            edges.append((gates[g][0], g+Ni))
            edges.append((gates[g][1], g+Ni))
        DG.add_edges_from(edges)
        return DG

    def connected_ground_truth(self, net):
        Ni = net.Ni
        No = net.No
        Ng = net.Ng
        digraph = self.graph_from_gates(net.gates, Ni, No)
        connected = chain.from_iterable(
            nx.ancestors(digraph, Ni+Ng-o-1) for o in range(No))
        connected = np.unique(list(connected))
        connected = np.union1d(connected, np.arange(Ng-No+Ni, Ng+Ni))
        result = np.zeros(Ng+Ni, dtype=np.uint8)
        result[connected] = 1
        return result

    def assert_feedforward(self, net):
        for g in range(net.Ng):
            assert net.gates[g][0] < g + net.Ni
            assert net.gates[g][1] < g + net.Ni

    # ############## Fixtures ##################### #
    @fixture(scope='module')
    def any_network(self, network_file_instance):
        with open(network_file_instance) as f:
            net = harness_to_fixture(f)
        return net

    @fixture(scope='module')
    def random_network(self):
        # generate random feedforward network
        Ni = randint(1, 10)
        No = randint(1, 10)
        Ng = randint(No, 20)
        gates = [(randint(g+Ni), randint(g+Ni), randint(16))
                 for g in range(Ng)]
        # create the seed network
        return BoolNet(gates, Ni, No)

    # ############## Tests ##################### #
    def test_acyclic(self, random_network):
        DG = self.graph_from_gates(
            random_network.gates, random_network.Ni, random_network.No)
        assert nx.is_directed_acyclic_graph(DG)

    def test_feedforward(self, random_network):
        self.assert_feedforward(random_network)

    def test_connected_sources(self, random_network):
        expected = self.connected_ground_truth(random_network)
        actual = random_network.connected_sources()
        assert_array_equal(expected, actual)

    def test_connected_gates(self, random_network):
        Ni = random_network.Ni
        expected = self.connected_ground_truth(random_network)[Ni:]
        actual = random_network.connected_gates()
        assert_array_equal(expected, actual)

    def test_revert_move_revert(self, any_network):
        net = copy(any_network['net'])
        for i in range(20):
            # old_gates = copy(net.gates)
            old_gates = np.array(net.gates, copy=True)
            net.move_to_random_neighbour()
            assert not np.array_equal(net.gates, old_gates)
            net.revert_move()
            assert_array_equal(net.gates, old_gates)

    def test_revert_move_revert_multi(self, any_network):
        net = copy(any_network['net'])
        backups = []
        # move network several times keeping a list of the networks
        for i in range(20):
            backups.append(copy(net))
            net.move_to_random_neighbour()
        # revert it incrementally and check it is equal to the backup each time
        for backup in reversed(backups):
            net.revert_move()
            assert_array_equal(net.gates, backup.gates)

    def test_revert_all_moves(self, any_network):
        net = copy(any_network['net'])
        for i in range(10):
            # old_gates = copy(net.gates)
            old_gates = np.array(net.gates, copy=True)
            for i in range(10):
                net.move_to_random_neighbour()
            net.revert_all_moves()
            assert_array_equal(net.gates, old_gates)

    def test_multiple_move_gates(self, any_network):
        net = copy(any_network['net'])
        for i in range(20):
            # old_gates = copy(net.gates)
            old_gates = np.array(net.gates, copy=True)
            net.move_to_random_neighbour()
            assert not np.array_equal(net.gates, old_gates)

    @mark.parametrize('repeats', range(10))
    def test_move_in_connected_range(self, repeats, any_network):
        net = copy(any_network['net'])
        connected = self.connected_ground_truth(net)
        gate = net.random_move()['gate']
        assert connected[gate] == 1

    @mark.parametrize('repeats', range(3))
    def test_feedforward_after_randomise(self, repeats, any_network):
        net = copy(any_network['net'])
        net.randomise()
        self.assert_feedforward(net)

    @mark.parametrize('repeats', range(3))
    def test_shape_after_randomise(self, repeats, random_network):
        net = copy(random_network)
        expected = net.gates.shape
        net.randomise()
        assert net.gates.shape == expected

    @mark.parametrize('repeats', range(3))
    def test_different_after_randomise(self, repeats, random_network):
        net = copy(random_network)
        old_gates = np.array(net.gates)
        net.randomise()
        assert not np.array_equal(old_gates, net.gates)

    def test_max_depth(self, any_network):
        net = copy(any_network['net'])
        expected = any_network['max depths']
        actual = net.max_node_depths()
        assert_array_equal(actual, expected)

    def test_copy(self, any_network):
        net1 = copy(any_network['net'])
        net1.move_to_random_neighbour()

        net2 = copy(net1)
        assert_array_equal(net2.gates, net1.gates)
        assert net2.Ni == net1.Ni
        assert net2.No == net1.No
        assert net2.Ng == net1.Ng
        assert not net2.history_empty()
        net2.clear_history()
        assert net2.history_empty()
        assert not net1.history_empty()

        assert_array_equal(net2.connected_gates(), net1.connected_gates())
        assert_array_equal(net2.connected_sources(), net1.connected_sources())

    def test_set_gates_copies(self, any_network):
        net = copy(any_network['net'])
        gates = np.array(net.gates)
        net.set_gates(gates)
        gates += 1
        assert_array_less(net.gates, gates)
