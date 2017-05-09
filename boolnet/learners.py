from copy import copy
from collections import namedtuple
import numpy as np
import bitpacking.packing as pk
import minfs.feature_selection as mfs

import boolnet.bintools.functions as fn
from boolnet.utils import PackedMatrix, order_from_rank, inverse_permutation
from boolnet.network.networkstate import BNState
from time import time


LearnerResult = namedtuple('LearnerResult', [
    'network', 'partial_networks', 'best_errors', 'best_iterations',
    'final_iterations', 'target_order', 'feature_sets', 'restarts',
    'optimisation_time', 'other_time'])


def fn_value_stop_criterion(func_id, name, limit=None):
    if limit is None:
        limit = fn.optimum(func_id)
    if fn.is_minimiser(func_id):
        return lambda state: state.function_value(name) <= limit
    else:
        return lambda state: state.function_value(name) >= limit


class BasicLearner:
    def _setup(self, optimiser, parameters):
        # Gate generation
        self.gate_generator = parameters['gate_generator']
        self.node_funcs = parameters['network']['node_funcs']
        self.budget = parameters['network']['Ng']
        # Instance
        self.problem_matrix = parameters['training_set']
        self.Ni = self.problem_matrix.Ni
        self.No = self.problem_matrix.No
        self.input_matrix, self.target_matrix = np.split(
            self.problem_matrix, [self.Ni])
        self.Ne = self.problem_matrix.Ne
        # Optimiser
        self.optimiser = optimiser
        self.opt_params = copy(parameters['optimiser'])
        gf_name = self.opt_params['guiding_function']
        self.guiding_fn_id = fn.function_from_name(gf_name)
        self.guiding_fn_params = self.opt_params.get(
            'guiding_function_parameters', {})
        self.opt_params['minimise'] = fn.is_minimiser(self.guiding_fn_id)

        # convert shorthands for target order
        if parameters['target_order'] == 'lsb':
            self.target_order = np.arange(self.No, dtype=np.uintp)
        elif parameters['target_order'] == 'msb':
            self.target_order = np.arange(self.No, dtype=np.uintp)[::-1]
        elif parameters['target_order'] == 'random':
            self.target_order = np.random.permutation(self.No).astype(np.uintp)
        elif parameters['target_order'] == 'auto':
            self.target_order = None
            # this key is only required if auto-targetting
            self.mfs_metric = parameters['minfs_selection_metric']
        else:
            self.target_order = np.array(parameters['target_order'],
                                         dtype=np.uintp)
        # Optional minfs solver time limit
        self.minfs_params = parameters.get('minfs_solver_params', {})
        self.minfs_solver = parameters.get('minfs_solver', 'cplex')

        # add functor for evaluating the guiding func
        self.guiding_fn_eval_name = 'guiding'
        self.opt_params['guiding_function'] = lambda x: x.function_value(
            self.guiding_fn_eval_name)

        # Check if user supplied a stopping condition
        condition = self.opt_params.get('stopping_condition', None)
        if condition and condition[0] != 'guiding':
            limit = condition[1]
            self.stopping_fn_eval_name = 'stop'
            self.stopping_fn_id = fn.function_from_name(condition[0])
            if len(condition) > 2:
                self.stopping_fn_params = condition[2]
            else:
                self.stopping_fn_params = {}
        else:
            if condition:
                limit = condition[1]
            self.stopping_fn_eval_name = self.guiding_fn_eval_name
            self.stopping_fn_id = self.guiding_fn_id
            self.stopping_fn_params = self.guiding_fn_params
            limit = None

        self.opt_params['stopping_condition'] = fn_value_stop_criterion(
            self.stopping_fn_id, self.stopping_fn_eval_name, limit)
        # check parameters
        if self.guiding_fn_id not in fn.scalar_functions():
            raise ValueError('Invalid guiding function: {}'.format(gf_name))
        if max(self.node_funcs) > 15 or min(self.node_funcs) < 0:
            raise ValueError('\'node_funcs\' must come from [0, 15]: {}'.
                             format(self.node_funcs))

    def run(self, optimiser, parameters):
        t0 = time()
        self._setup(optimiser, parameters)

        if self.target_order is None:
            # determine the target order by ranking feature sets
            mfs_features = pk.unpackmat(self.input_matrix, self.Ne)
            mfs_targets = pk.unpackmat(self.target_matrix, self.Ne)

            # use external solver for minFS
            rank, feature_sets = mfs.ranked_feature_sets(
                mfs_features, mfs_targets, self.mfs_metric,
                self.minfs_solver, self.minfs_params)

            # randomly pick from possible exact orders
            self.target_order = order_from_rank(rank)

        # build the network state
        gates = self.gate_generator(self.budget, self.Ni,
                                    self.No, self.node_funcs)

        # reorder problem matrix
        outputs = self.problem_matrix[-self.No:, :]
        outputs[:] = outputs[self.target_order, :]

        state = BNState(gates, self.problem_matrix)
        # add the guiding function to be evaluated
        state.add_function(self.guiding_fn_id, self.guiding_fn_eval_name,
                           self.guiding_fn_params)
        # add the stopping function to be evaluated
        if (self.stopping_fn_eval_name is not None and
                self.stopping_fn_eval_name != self.guiding_fn_eval_name):
            state.add_function(self.stopping_fn_id, self.stopping_fn_eval_name,
                               self.stopping_fn_params)

        t1 = time()
        # run the optimiser
        opt_result = self.optimiser.run(state, self.opt_params)
        t2 = time()

        # undo ordering
        inverse_order = inverse_permutation(self.target_order)
        outputs[:] = outputs[inverse_order, :]

        gates = np.array(opt_result.representation.gates)
        out_gates = gates[-self.No:, :]
        out_gates[:] = out_gates[inverse_order, :]
        opt_result.representation.set_gates(gates)

        return LearnerResult(
            network=opt_result.representation,
            best_errors=[opt_result.error],
            best_iterations=[opt_result.best_iteration],
            final_iterations=[opt_result.iteration],
            restarts=[opt_result.restarts],
            target_order=self.target_order,
            feature_sets=None,
            partial_networks=[],
            optimisation_time=t2-t1,
            other_time=t1-t0)
