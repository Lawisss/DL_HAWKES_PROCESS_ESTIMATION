import io
import os
from setuptools import setup, find_packages


def read(*paths, **kwargs):
    
    """
    Read the contents of a text file safely
    
    >>> read("project_name", "VERSION")
    '0.1.0'
    >>> read("README.md")
    ...
    
    """
    
    content = ""
    
    with io.open(os.path.join(os.path.dirname(__file__), *paths), encoding=kwargs.get("encoding", "utf8"),) as open_file:
        content = open_file.read().strip()
        
    return content


def read_requirements(path):
    return [line.strip() for line in read(path).split("\n") if not line.startswith(('"', "#", "-", "git+"))]

setup(
    name='DL_HAWKES_PROCESS_ESTIMATION',
    version='1.0',
    description='DL & HAWKES PROCESS ESTIMATION',
    author='Nicolas Girard',
    author_email='nico.girard22@gmail.com',
    url='https://github.com/Lawisss',
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    keywords='development, setup, setuptools',
    package_dir={'': 'SRC'},
    packages=find_packages(where='SRC'),
    python_requires='>=3.7',
    install_requires=read_requirements("requirements.txt"),
    extras_require={'mpi': ['mpi4py'], 'test': ['pytest', 'pytest-cov', 'codecov']},
    scripts=[],
    entry_points={},
    classifiers=['Development Status :: 3 - Alpha',
                 'Environment :: Console',
                 'Environment :: Win32 (MS Windows)',
                 'Intended Audience :: Developers',
                 'Intended Audience :: Science/Research',
                 'Intended Audience :: Information Technology',
                 'Intended Audience :: Financial and Insurance Industry',
                 'License :: OSI Approved :: MIT License',
                 'Natural Language :: English',
                 'Operating System :: Microsoft :: Windows',
                 'Programming Language :: Python :: 3.11',
                 'Topic :: Office/Business :: Financial :: Investment',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence']
)
