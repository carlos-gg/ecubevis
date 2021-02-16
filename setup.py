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
    #url='https://earth.bsc.es/gitlab/cgome1/ecubevis',
    keywords=[
        'visualization', 
        'interactive', 
        'plotting', 
        'earth-data', 
        'package'
        ],
    install_requires=[
        'numpy ~= 1.18',
        'matplotlib >= 2.2',
        'xarray == 0.15.1',
        'hvplot == 0.5.2',
        'holoviews == 1.13.2',
        'bokeh == 2.0.1',
        'pyviz_comms == 0.7.4'
        ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        #'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux',
        'Natural Language :: English',
        #'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering'
        ],
)