import yaml
import boolnet.exptools.config_filtering as cf


config = yaml.load('''---
name: varNe_add4_all_short

seeding:
  sampling: 'shared'
  learner: 'unique'

# common params
data:
  type: generated
  operator: add
  bits: 4

sampling:
  type: generated
  Ns: 20

learner:
  name: basic
  target_order: lsb
  network:
    method: generated
    Ng: 84
    node_funcs: [7]
  optimiser:
    name:             LAHC
    cost_list_length: 1000
    max_iterations:   8000000
    max_restarts:     20
...''')


log_keys = [
    ('learner', True,  ['learner', 'name']),
    ('config_num', True,  ['configuration_number']),
    ('trg_set_num', True,  ['training_set_number']),
    ('tfs', True,  ['learner', 'network', 'node_funcs']),
    ('trg_indices', True,  ['mapping', 'training_indices']),
    ('guiding_function', True,  ['learner', 'optimiser', 'guiding_function']),
    ('fs_sel_metric', False, ['learner', 'minfs_selection_metric']),
    ('fs_masking', False, ['learner', 'minfs_masking']),
    ('given_tgt_order', False, ['learner', 'target_order']),
    ('opt_{}', False, ['learner', 'optimiser', '.*']),
]
log_keys_just_paths = [(entry[0], entry[2]) for entry in log_keys]


def test_path_value_pairs():
    expected = sorted([
        (['data', 'bits'], 4),
        (['data', 'operator'], 'add'),
        (['data', 'type'], 'generated'),
        (['seeding', 'sampling'], 'shared'),
        (['seeding', 'learner'], 'unique'),
        (['sampling', 'Ns'], 20),
        (['sampling', 'type'], 'generated'),
        (['learner', 'network', 'Ng'], 84),
        (['learner', 'network', 'method'], 'generated'),
        (['learner', 'network', 'node_funcs'], [7]),
        (['learner', 'target_order'], 'lsb'),
        (['learner', 'optimiser', 'max_iterations'], 8000000),
        (['learner', 'optimiser', 'cost_list_length'], 1000),
        (['learner', 'optimiser', 'max_restarts'], 20),
        (['learner', 'optimiser', 'name'], 'LAHC'),
        (['learner', 'name'], 'basic'),
        (['name'], 'varNe_add4_all_short')])
    actual = sorted(cf.path_value_pairs(config))
    assert expected == actual


def test_list_regex_match():
    assert cf.list_regex_match([], [])
    assert not cf.list_regex_match([''], [])
    assert not cf.list_regex_match([], [''])
    assert cf.list_regex_match(['seeding', 'sampling'], ['seeding', 'sampling'])
    assert cf.list_regex_match(['seed.*', 'sampling'], ['seeding', 'sampling'])
    assert cf.list_regex_match(['seeding', '.*ling'], ['seeding', 'sampling'])
    assert cf.list_regex_match(['seed.*', '.*ling'], ['seeding', 'sampling'])
    assert cf.list_regex_match(['learner', 'optimiser', '.*'],
                               ['learner', 'optimiser', 'name'])


def test_filter_keys():
    expected = {'given_tgt_order': 'lsb',
                'learner': 'basic',
                'tfs': [7],
                'opt_name': 'LAHC',
                'opt_cost_list_length': 1000,
                'opt_max_iterations': 8000000,
                'opt_max_restarts': 20}
    actual = cf.filter_keys(config, log_keys_just_paths)
    assert expected == actual
