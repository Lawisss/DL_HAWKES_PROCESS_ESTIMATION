#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""CLI module

File running main executive function (app.py) with CLI commands

"""

import argparse

from app import main
import variables.parser_var as prs

# Arguments parser function

def argparser() -> None:

    """
    CLI arguments parser function

    Args:
        None: Function contain no arguments

    Returns:
        None: Function does not return anything

    """

    # Created an argument parser
    parser = argparse.ArgumentParser()
                
    # Iterated over filtered argument groups
    for group in prs.ARG_GROUPS:
        group_args = parser.add_argument_group(group['name'], group['description'])

        # Iterated over each argument in group
        for arg in group['args']:
            group_args.add_argument(arg['name'], **{k: v for k, v in arg.items() if k in ('type', 'nargs', 'default', 'help')})

    # Parsed arguments from command line 
    args, _ = parser.parse_known_args()
    
    return args

if __name__ == "__main__":

    args = argparser()
    main(args)