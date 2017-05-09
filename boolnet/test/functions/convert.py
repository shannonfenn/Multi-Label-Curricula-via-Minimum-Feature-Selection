import json
import argparse
import os.path
import numpy as np


def main(in_name):
    out_name = os.path.splitext(in_name)[0]

    with open(in_name) as f:
        ds_settings = json.load(f)

    func = np.array(ds_settings['function'])

    inps = np.asarray(func[:, 0].tolist(), dtype=np.uint8)
    tgts = np.asarray(func[:, 1].tolist(), dtype=np.uint8)

    np.savez(out_name, input_matrix=inps, target_matrix=tgts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert old json dataset to numpy binary file.')
    parser.add_argument('filename', type=str,
                        help='name of file to convert (should be json formatted)')
    args = parser.parse_args()

    main(args.filename)
