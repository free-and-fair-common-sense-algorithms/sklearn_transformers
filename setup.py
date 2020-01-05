from setuptools import setup
from setuptools import find_packages

setup(
    name='sklearn_transformers',
    version='0.1',
    packages=find_packages(),
    author='brian clifton',
    install_requires=['pandas', 'numpy', 'sklearn']
)
