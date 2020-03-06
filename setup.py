#!/usr/bin/env python

from setuptools import setup, find_packages
import subprocess
import os


def write_version_file(version):
    """ Writes a file with version information to be used at run time

    Parameters
    ----------
    version: str
        A string containing the current version information

    Returns
    -------
    version_file: str
        A path to the version file

    """
    try:
        git_log = subprocess.check_output(
            ['git', 'log', '-1', '--pretty=%h %ai']).decode('utf-8')
        git_diff = (subprocess.check_output(['git', 'diff', '.']) +
                    subprocess.check_output(
                        ['git', 'diff', '--cached', '.'])).decode('utf-8')
        if git_diff == '':
            git_status = '(CLEAN) ' + git_log
        else:
            git_status = '(UNCLEAN) ' + git_log
    except Exception as e:
        print("Unable to obtain git version information, exception: {}"
              .format(e))
        git_status = ''

    version_file = '.version'
    if os.path.isfile(version_file) is False:
        with open('dynamical_population_models/' + version_file, 'w+') as f:
            f.write('{}: {}'.format(version, git_status))

    return version_file


def get_long_description():
    """ Finds the README and reads in the description """
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, 'README.md')) as f:
        long_description = f.read()
    return long_description


# get version info from __init__.py
def readfile(filename):
    with open(filename) as fp:
        filecontents = fp.read()
    return filecontents


VERSION = '0.0.1'
version_file = write_version_file(VERSION)
long_description = get_long_description()

setup(name='dynamical_population_models',
      description='Population inference models for binaries in globular clusters',
      long_description=long_description,
      url='https://github.com/ColmTalbot/dynamical_population_models',
      author='Colm Talbot',
      author_email='colm.talbot@monash.edu',
      license="MIT",
      version=VERSION,
      packages=find_packages(exclude=["test"]),
      package_dir={'dynamical_population_models': 'dynamical_population_models'},
      package_data={'dynamical_population_models': [version_file, "grid_dict_5e5",
                                                    "grid_dict_1e7","grid_dict_1e8"]},
      install_requires=['bilby', 'gwpopulation', 'numpy>=1.16'],
      classifiers=[
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent"])
