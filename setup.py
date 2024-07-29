import setuptools
from setuptools import setup

__version__ = "3.2.1"
install_deps = [
    "hmrGC @ git+https://github.com/BMRRgroup/fieldmapping-hmrGC.git"
]

setup(
    name='hmrGC_dualEcho',
    description='dual-echo water-fat seperation using hierarchical multi-resolution graph-cuts',
    author='Jonathan Stelter',
    author_email='jonathan.stelter@tum.de',
    packages=setuptools.find_packages(),
    version = __version__,
    install_requires=install_deps,
    package_data={'': ['*.json']},
)
