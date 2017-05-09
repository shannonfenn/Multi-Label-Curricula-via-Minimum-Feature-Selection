import json
import argparse
import os.path
import numpy as np


def unique_rows(data):
    uniq = np.unique(data.view(data.dtype.descr * data.shape[1]))
    return uniq.view(data.dtype).reshape(-1, data.shape[1])


def main(in_name):
    with np.load(in_name) as data:
        inputs = data['input_matrix']
        targets = data['target_matrix']

    # check input and target are non-empty
    assert inputs.shape[0] != 0
    assert inputs.shape[1] != 0
    assert targets.shape[0] != 0
    assert targets.shape[1] != 0
    # check input and target have same Ne
    assert inputs.shape[0] == targets.shape[0]
    # check no input patterns are repeated
    assert inputs.shape[0] == unique_rows(inputs).shape[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Validate dataset in numpy binary file.')
    parser.add_argument('filename', type=str,
                        help='name of file to check (should be npz formatted)')
    args = parser.parse_args()

    main(args.filename)
