# !/usr/bin/env python

import os
import re
from setuptools import setup


def resource(*args):
    return os.path.join(os.path.abspath(os.path.join(__file__, os.pardir)),
                        *args)


with open(resource('ecubevis', '__init__.py')) as version_file:
    version_file = version_file.read()
    VERSION = re.search(r"""^__version__ = ['"]([^'"]*)['"]""",
                        version_file, re.M)
    VERSION = VERSION.group(1)

with open(resource('README.md')) as readme_file:
    README = readme_file.read()

setup(
    name='ecubevis',
    packages=['ecubevis'],
    version=VERSION,
    description='Earth CUBE VISualization with Python',
    long_description=README,
    long_description_content_type='text/markdown',
    author='Carlos Alberto Gomez Gonzalez',
    #license='MIT',
    author_email='carlos.gomez@bsc.es',
    url='https://github.com/carlgogo/ecubevis',
    keywords=[
        'visualization', 
        'interactive', 
        'plotting', 
        'earth-data', 
        ],
    install_requires=[
        'numpy',
        'matplotlib',
        'xarray',
        'hvplot',
        'holoviews',
        'pyproj',
        'cartopy',
        'geoviews',
        'bokeh',
        'pyviz_comms',
        'joblib'
        ],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Natural Language :: English',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Visualization'
        ],
)