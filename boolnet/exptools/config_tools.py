from copy import deepcopy
from collections import MutableMapping
from progress.bar import Bar
import voluptuous as vol
import numpy as np
import os
import json
import random

import boolnet.exptools.config_schemata as sch
import boolnet.bintools.operator_iterator as op
from boolnet.utils import PackedMatrix


def get_seed(key):
    ''' Keeps a registry of seeds for each key, if given a new
        key get_seed() generates a new seed for that key, but if
        given an existing key it returns that seed. Allows any number
        of named seeds.'''
    if 'registry' not in get_seed.__dict__:
        # first call, create the registry
        get_seed.registry = {}
    if key not in get_seed.registry:
        # non-existant key, generate a seed
        random.seed()  # use default randomness source to get a seed
        get_seed.registry[key] = random.randint(1, 2**32-1)
    return get_seed.registry[key]


# to signal schema validation failure
# (with custom message formatting)
class ValidationError(Exception):
    pass


class ExperimentJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def dump_results_partial(results, stream, first):
    if not first:
        stream.write(',')
    json.dump(results, stream, cls=ExperimentJSONEncoder)
    stream.write('\n')
    # ensure data is written to disk immediately
    stream.flush()
    os.fsync(stream.fileno())


def update_nested(d, u):
    ''' this updates a dict with another where the two may contain nested
        dicts themselves (or more generally nested mutable mappings). '''
    for k, v in u.items():
        if isinstance(v, MutableMapping):
            r = update_nested(d.get(k, {}), v)
            d[k] = r
        else:
            # preference to second mapping if k exists in d
            d[k] = u[k]
    return d


def build_filename(params, extension, key='filename'):
    ''' complicated filename helper.'''
    filename = params[key]
    location = params.get('dir', None)
    # if 'filename' is absolute then ignore 'dir'
    if location and not os.path.isabs(params[key]):
        filename = os.path.join(location, params[key])
    # add extension if missing
    if os.path.splitext(filename)[-1] == '':
        filename += extension
    return filename


def load_samples(params, N, Ni, Nt=None):
    if params['type'] == 'given':
        training_indices = np.array(params['indices'], dtype=np.uintp)
        if 'test' in params:
            test_indices = np.array(params['test'], dtype=np.uintp)
        else:
            test_indices = [None]*training_indices.shape[0]

    if params['type'] == 'file':
        filename = build_filename(params, '.npy')
        training_indices = np.load(filename)
        if 'test' in params:
            filename = build_filename(params, '.npy', 'test')
            test_indices = np.load(filename)
        else:
            test_indices = [None]*training_indices.shape[0]

    elif params['type'] == 'generated':
        # this provided seed allows us to generate the same set
        # of training indices across multiple configurations
        Ns = params['Ns']
        Ne = params['Ne']
        s = params['seed']
        if isinstance(s, str):
            # s is actually a name
            s = get_seed(s)
            params['seed'] = s
        random.seed(s)
        if 'test' in params:
            Ne_test = params['test']
            if Nt is None:
                # combined training and test data
                all_indices = np.array([
                    random.sample(range(N), Ne+Ne_test) for i in range(Ns)])
                training_indices, test_indices = np.hsplit(all_indices, [Ne])
            else:
                # separate training and test data
                training_indices = np.array([
                    random.sample(range(N), Ne) for i in range(Ns)])
                test_indices = np.array([
                    random.sample(range(N), Ne_test) for i in range(Ns)])
        else:
            training_indices = np.array([
                random.sample(range(N), Ne) for i in range(Ns)])
            test_indices = [None]*Ns
    return training_indices, test_indices


def load_dataset(settings):
    data_settings = settings['data']

    dtype = data_settings['type']

    if dtype == 'file':
        instance, N, Ni = file_instance(data_settings)
    elif dtype == 'split':
        instance, N, Ni, Nt = split_instance(data_settings)
    elif dtype == 'generated':
        instance, N, Ni = generated_instance(data_settings)

    # check for problematic case
    problematic = (dtype == 'split' and
                   settings['sampling']['type'] == 'blank' and
                   'test' not in settings['sampling'])
    if problematic:
        raise ValueError('Cannot use implicit test sampling with split data.')

    # Only handle sampling if necessary
    if dtype == 'split':
        if settings['sampling']['type'] == 'blank':
            return [instance]
        else:
            training_indices, test_indices = load_samples(
                settings['sampling'], N, Ni, Nt)
    else:
        training_indices, test_indices = load_samples(
            settings['sampling'], N, Ni)

    contexts = []
    for trg, test in zip(training_indices, test_indices):
        context = instance.copy()
        context.update({'training_indices': trg, 'test_indices': test})
        contexts.append(context)
    return contexts


def file_instance(params):
    filename = build_filename(params, '.npz')
    with np.load(filename) as dataset:
        Mp = dataset['matrix']
        # Ne in the dataset is the full number of examples which we are
        # referring to herein as 'N' to differentiate from the sample size
        N = dataset['Ne']
        Ni = dataset['Ni']

    if 'targets' in params:
        Tp = Mp[Ni:, :]
        Tp = Tp[params['targets'], :]
        Mp = np.vstack((Mp[:Ni, :], Tp))

    instance = {
        'type': 'raw_unsplit',
        'matrix': PackedMatrix(Mp, N, Ni)
        }

    return instance, N, Ni


def split_instance(params):
    trg_filename = build_filename(params, '.npz', key='training_filename')
    test_filename = build_filename(params, '.npz', key='test_filename')
    with np.load(trg_filename) as train, np.load(test_filename) as test:
        Mp_trg = train['matrix']
        Mp_test = test['matrix']
        
        Ne_trg, Ne_test = train['Ne'], test['Ne']
        Ni = train['Ni']
        assert test['Ni'] == Ni

        if 'targets' in params:
            Tp_trg = Mp_trg[Ni:, :]
            Tp_trg = Tp_trg[params['targets'], :]
            Mp_trg = np.vstack((Mp_trg[:Ni, :], Tp_trg))
            Tp_test = Mp_test[Ni:, :]
            Tp_test = Tp_test[params['targets'], :]
            Mp_test = np.vstack((Mp_test[:Ni, :], Tp_test))

        instance = {
            'type': 'raw_split',
            'training_set': PackedMatrix(Mp_trg, Ne_trg, Ni),
            'test_set': PackedMatrix(Mp_test, Ne_test, Ni)
            }

    return instance, Ne_trg, Ni, Ne_test


def generated_instance(params):
    Nb = params['bits']
    operator = op.operator_from_name(params['operator'])

    instance = {
        'type': 'operator',
        'operator': operator,
        'Nb': Nb,
        'No': params.get('out_width', Nb),  # defaults to operand width
        'window_size': params.get('window_size', 4)  # arbitrary default
    }
    Ni = op.num_operands(operator) * Nb

    return instance, 2**Ni, Ni


# def handle_initial_network(settings):
#     net_settings = settings['learner']['network']
#     net_method = net_settings['method']
#     if net_method == 'given':
#         data_dir = settings['data']['dir']
#         filename = net_settings['file']
#         index = net_settings['index']
#         with open(join(data_dir, filename)) as f:
#             gates = np.array(json.load(f)[index], dtype=np.uint32)
#             net_settings['initial_gates'] = gates


def validate_schema(config, schema, config_num, msg):
    try:
        schema(config)
    except vol.MultipleInvalid as err:
        msg = ('Experiment instance {} invalid: {}\nerror: {}\npath: {}\n'
               '\nConfig generation aborted.').format(
            config_num + 1, err, err.error_message, err.path)
        raise ValidationError(msg)


def split_variables_from_base(settings):
    # configuration sub-dicts are popped
    try:
        variable_sets = settings['list']
    except KeyError:
        try:
            products = settings['product']
            # build merged mappings for each pair from products
            variable_sets = [update_nested(deepcopy(d1), d2)
                             for d2 in products[1]
                             for d1 in products[0]]
        except KeyError:
            print('Warning: no variable configuration found.\n')
            variable_sets = [{}]

    return variable_sets, settings['base_config']


def insert_default_log_keys(settings):
    defaults = [
        ['learner', True, ['learner', 'name']],
        ['learner_seed', True, ['learner', 'seed']],
        ['config_num', True, ['configuration_number']],
        ['trg_set_num', True, ['training_set_number']],
        ['tfs', True, ['learner', 'network', 'node_funcs']],
        ['guiding_function', True, ['learner', 'optimiser',
                                    'guiding_function']],
        ['given_tgt_order', True, ['learner', 'target_order']],
        ['sample_seed', False, ['sampling', 'seed']],
        ['fs_sel_metric', False, ['learner', 'minfs_selection_metric']],
        ['fs_masking', False, ['learner', 'minfs_masking']],
        ['fs_timelimit', False, ['learner', 'minfs_time_limit']],
        ['fs_solver', False, ['learner', 'minfs_solver']],
        ['opt_{}', False, ['learner', 'optimiser', '.*']],
    ]
    # give preference to user log_keys
    defaults.extend(settings['base_config'].get('log_keys', []))
    settings['base_config']['log_keys'] = defaults
    return settings


def generate_configurations(settings, batch_mode):
    # validate the given schema
    try:
        sch.experiment_schema(settings)
    except vol.MultipleInvalid as err:
        raise ValidationError(
            'Top-level config invalid: {}\nerror: {}\npath: {}'.format(
                err, err.error_message, err.path))

    # insert default log_keys values into base config
    insert_default_log_keys(settings)

    # the configurations approach involves having a multiple config dicts and
    # updating them with each element of the configurations list or product
    variable_sets, base_settings = split_variables_from_base(settings)

    # Build up the configuration list
    configurations = []

    if not batch_mode:
        bar = Bar('Generating configurations', max=len(variable_sets),
                  suffix='%(index)d/%(max)d : %(eta)ds')
        bar.update()
    try:
        for config_num, variables in enumerate(variable_sets):
            # keep contexts isolated
            context = deepcopy(base_settings)
            # update the settings dict with the values for this configuration
            update_nested(context, variables)
            # check the given config is a valid experiment
            validate_schema(context, sch.instance_schema,
                            config_num, variables)
            # record the config number for debugging
            context['configuration_number'] = config_num
            # !!REMOVED!! load initial network from file if required
            # handle_initial_network(context)
            # load the data for this configuration
            instances = load_dataset(context)

            configurations.append((context, instances))
            if not batch_mode:
                bar.next()
    finally:
        # clean up progress bar before printing anything else
        if not batch_mode:
            bar.finish()
    return configurations


def generate_tasks(configurations, batch_mode):
    # Build up the task list
    tasks = []

    if not batch_mode:
        bar = Bar('Generating training tasks', max=len(configurations),
                  suffix='%(index)d/%(max)d : %(eta)ds')
        bar.update()
    try:
        for context, instances in configurations:
            # for each sample
            for i, instance in enumerate(instances):
                task = deepcopy(context)
                task['mapping'] = instance
                task['training_set_number'] = i
                tasks.append(task)
            if not batch_mode:
                bar.next()
    finally:
        # clean up progress bar before printing anything else
        if not batch_mode:
            bar.finish()
    return tasks
