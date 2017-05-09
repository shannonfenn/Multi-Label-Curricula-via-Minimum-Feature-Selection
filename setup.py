try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from Cython.Distutils import Extension, build_ext
import numpy as np
import cython_gsl
import sys
import glob
import os

include_dirs = [np.get_include(),
                cython_gsl.get_cython_include_dir()]

args = sys.argv[1:]

# get rid of intermediate and library files
if "clean" in args:
    print("Deleting cython files...")
    to_remove = []
    # C
    to_remove += glob.glob('boolnet/bintools/functions.c')
    to_remove += glob.glob('boolnet/bintools/biterror.c')
    to_remove += glob.glob('boolnet/bintools/operator_iterator.c')
    to_remove += glob.glob('boolnet/bintools/example_generator.c')
    to_remove += glob.glob('boolnet/exptools/fastrand.c')
    # C++
    to_remove += glob.glob('boolnet/network/algorithms.cpp')
    to_remove += glob.glob('boolnet/network/boolnet.cpp')
    to_remove += glob.glob('boolnet/network/networkstate.cpp')
    # Static lib files
    to_remove += glob.glob('boolnet/bintools/*.so')
    to_remove += glob.glob('boolnet/exptools/*.so')
    to_remove += glob.glob('boolnet/network/*.so')
    for f in to_remove:
        os.remove(f)


# We want to always use build_ext --inplace
if args.count('build_ext') > 0 and args.count('--inplace') == 0:
    sys.argv.insert(sys.argv.index('build_ext')+1, '--inplace')

extensions = [
    Extension('boolnet.bintools.functions',
              ['boolnet/bintools/functions.pyx'],
              include_dirs=include_dirs),
    Extension('boolnet.bintools.biterror',
              ['boolnet/bintools/biterror.pyx'],
              include_dirs=include_dirs),
    Extension('boolnet.bintools.operator_iterator',
              ['boolnet/bintools/operator_iterator.pyx'],
              include_dirs=include_dirs),
    Extension('boolnet.bintools.example_generator',
              ['boolnet/bintools/example_generator.pyx'],
              include_dirs=include_dirs),
    Extension('boolnet.exptools.fastrand',
              ['boolnet/exptools/fastrand.pyx'],
              libraries=cython_gsl.get_libraries(),
              library_dirs=[cython_gsl.get_library_dir()],
              include_dirs=include_dirs),
    # C++
    Extension('boolnet.network.algorithms',
              ['boolnet/network/algorithms.pyx'],
              language='c++',
              include_dirs=include_dirs),
    Extension('boolnet.network.boolnet',
              ['boolnet/network/boolnet.pyx'],
              language='c++',
              include_dirs=include_dirs),
    Extension('boolnet.network.networkstate',
              ['boolnet/network/networkstate.pyx'],
              language='c++',
              include_dirs=include_dirs),
    ]

setup(
    name='boolnet',
    include_dirs=[np.get_include(), cython_gsl.get_include()],
    cmdclass={'build_ext': build_ext},
    ext_modules=extensions
    )
