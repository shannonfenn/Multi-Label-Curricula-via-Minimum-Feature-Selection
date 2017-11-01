Code and data for the paper "Multi-Label Curricula via Minimum Feature Selection: a Case Study in Boolean Networks"

## Setup

To install requirements from PYPI run:
~~~~
pip install -r requirements.txt
~~~~

Depends on two other projects [bitpacking (v1.0)](https://github.com/shannonfenn/bitpacking/releases/tag/v1.0) and [minfs (v1.0)](https://github.com/shannonfenn/minfs/releases/tag/v1.0) which need to be available on the python path.

## Use

To run use:

`python runexp.y <path-to-experiment-config>`

The experiments are implemented as yaml configs, see config_schema.py for the schema definition (note that the configs include relative paths to their relevant dataset and sample files, these may need to be updated depending on your directory structure).

Other commandline parameters can be given, use `python runexp.y -h` to see more.

Config files can be tested without running the actual experiment using:

`python check_config.y <path-to-experiment-config>`

which has similar arguments.


## Tests

Unit tests are implemented using pytest:
~~~~
pip install pytest
py.test -x
~~~~
