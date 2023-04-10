#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Discretisation MPI module

File containing parallelized Aggregated Hawkes Process functions (Hawkes Process discrete conversion).

"""

import numpy as np
from functools import partial

import VARIABLES.variables as var
from UTILS.utils import write_csv