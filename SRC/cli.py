#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""CLI module

File running main() executive function with CLI commands

"""

from UTILS.utils import argparser
from app import main

if __name__ == "__main__":

    args = argparser()
    main(args)