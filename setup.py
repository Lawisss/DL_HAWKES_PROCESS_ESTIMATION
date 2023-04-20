from setuptools import setup, find_packages

setup(
    name='DL_HAWKES_PROCESS_ESTIMATION',
    version='1.0',
    description='DL & HAWKES PROCESS ESTIMATION',
    author='Nicolas Girard',
    author_email='nico.girard22@gmail.com',
    packages=find_packages(),
    install_requires=['numpy', 'pandas', 'hawkes', 'torch>=1.9.0', 
                      'tensorflow>=2.0', 'tqdm', 'torchinfo'],
    extras_require={'mpi': ['mpi4py'], 'test': ['pytest', 'pytest-cov', 'codecov']},
    classifiers=['Development Status :: 3 - Alpha', 'Programming Language :: Python :: 3.8',
                 'License :: OSI Approved :: MIT License', 'Operating System :: OS Independent'],
)