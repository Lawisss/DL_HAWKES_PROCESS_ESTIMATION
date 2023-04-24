#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Parser arguments module

File containing parser function for command-line arguments

"""

import argparse
import functools
from typing import List, Callable

import VARIABLES.parser_var as prs


# Command-line argument parser function

def argparser(func: Callable = None, parse_args: bool = False, arg_groups: List[str] = None):

    """
    Command-line argument parser decorator

    Args:
        func (Callable): Function to be decorated
        parse_args (bool): If True, parses arguments from command line
        arg_groups (List[str]): List of argument group to include. If None, all argument groups are included

    Returns:
        Callable: Decorated function
    """

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            # Created an argument parser
            parser = argparse.ArgumentParser(prog=prs.PROG, usage=prs.USAGE, description=prs.DESCRIPTION)

            # Filtered argument groups based on whether the arg_groups parameter is set or not
            filtered_groups = filter(lambda grp: grp['name'] in arg_groups, prs.ARG_GROUPS) if arg_groups else prs.ARG_GROUPS
            
            # Iterated over filtered argument groups
            for group in filtered_groups:
                group_args = parser.add_argument_group(group['name'], group['description'])

                # Iterated over each argument in group
                for arg in group['args']:
                     kwargs = {key: arg[key] for key in ('type', 'default', 'help') if key in arg}
                     kwargs['nargs'] = arg['nargs']
                     group_args.add_argument(arg['name'], **kwargs)

            # If parse_args, parsed arguments from command line 
            args_parsed = parser.parse_args() if parse_args else None

            return func(args_parsed, *args, **kwargs)
        
        return wrapper

    # Decorator call, ex: @argparser / Factory call, ex: @argparser()
    return decorator(func) if func else decorator


        
    