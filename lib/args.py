# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import argparse
import glob
import os
import socket
import sys
import textwrap
from enum import Enum

from nsys_recipe.lib import recipe
from nsys_recipe.log import logger

def range_type(value):
    rangelist = []
    for valuei in value.split(','):
        try:
            start, end = map(float, valuei.split('-'))
            #start = start * 1e9
            #end = end * 1e9
            if start > end:
                raise argparse.ArgumentTypeError(f"Invalid range: {valuei}")
            rangelist.append((start, end))
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid range format: {valuei}")
    return rangelist

def _replace_range(name, start_index, end_index, value):
    return name[:start_index] + str(value) + name[end_index + 1 :]


def _substitute_env_var(name, index):
    pos = index + 1
    if pos >= len(name) or name[pos] != "{":
        logger.error("Missing '{' token after '%q' expression.")
        return name, len(name)

    pos += 1
    end = name.find("}", pos)
    if end == -1:
        logger.error("Missing '}' token after '%q{' expression.")
        return name, len(name)

    env_var = name[pos:end]
    value = os.getenv(env_var)

    if value is None:
        logger.warning(f"Environment variable '{env_var}' is not set.")
        return name, end + 1

    start = index - 1
    return _replace_range(name, start, end, value), start + len(value)


def _substitute_hostname(name, index):
    try:
        hostname = socket.gethostname()
    except socket.error as e:
        logger.warning(f"Unable to get host name: {e}")
        hostname = ""

    start = index - 1
    end = start + 1
    return _replace_range(name, start, end, hostname), start + len(hostname)


def _substitute_pid(name, index):
    pid = os.getpid()
    start = index - 1
    end = start + 1
    return _replace_range(name, start, end, pid), start + len(str(pid))


def _substitute_counters(name, counter_indices):
    if not counter_indices:
        return name

    orig_name = name
    for num in range(1, sys.maxsize):
        name = orig_name
        for index in reversed(counter_indices):
            name = _replace_range(name, index, index + 1, num)
        if not os.path.exists(name):
            return name
        num += 1

    raise ValueError("Maximum limit reached. Unable to find an available output name.")


def process_output(name):
    counter_indices = []

    index = name.find("%")
    while index != -1:
        index += 1
        if index >= len(name):
            logger.error("Unterminated " % " expression.")
            return name

        token = name[index]
        if token == "q":
            name, index = _substitute_env_var(name, index)
        elif token == "h":
            name, index = _substitute_hostname(name, index)
        elif token == "p":
            name, index = _substitute_pid(name, index)
        elif token == "n":
            counter_indices.append(index - 1)
        elif token == "%":
            name = _replace_range(name, index, index, "")
        else:
            logger.error(f"Unknown expression '%{token}'.")

        index = name.find("%", index)

    return _substitute_counters(name, counter_indices)


def process_directory(report_dir):
    files = []

    report_dir = os.path.abspath(report_dir)
    for ext in ("*.nsys-rep", "*.qdrep"):
        files.extend(glob.glob(os.path.join(report_dir, ext)))

    if not files:
        raise argparse.ArgumentTypeError("No nsys-rep files found.")

    return files


def process_input(path):
    extensions = (".qdrep", ".nsys-rep")
    n = None

    if ":" in path and not os.path.exists(path):
        path, n = path.rsplit(":", 1)
        try:
            n = int(n)
        except ValueError as e:
            raise argparse.ArgumentTypeError(
                "Expecting an integer value after the colon in the path."
            ) from e

    if os.path.isfile(path):
        if n is not None:
            raise argparse.ArgumentTypeError(
                "The ':n' syntax cannot be used for files."
            )
        if not path.endswith(extensions):
            raise argparse.ArgumentTypeError(f"{path} is not a nsys-rep file.")
        return path

    if os.path.isdir(path):
        files = sorted(
            file
            for extension in extensions
            for file in glob.glob(os.path.join(path, f"*{extension}"))
        )
        if not files:
            raise argparse.ArgumentTypeError(f"{path} does not contain nsys-rep files.")
        return files[:n]

    raise argparse.ArgumentTypeError(f"{path} does not exist.")


def process_integer(min_value):
    """Type function for argparse
    Returns a function that takes only a string argument and checks if the provided argument is
    greater than the min_value. Otherwise it raises an exception.
    The reason for this structure of functions is 'The argument to type can be any callable that
    accepts a single string.'"""

    def type_function(number_str):
        try:
            number = int(number_str)
        except ValueError:
            raise argparse.ArgumentTypeError("The argument must be an integer number")
        if number < min_value:
            raise argparse.ArgumentTypeError(
                f"The argument must be greater or equal to {min_value}"
            )
        return number

    return type_function


class TextHelpFormatter(argparse.HelpFormatter):
    """This class is similar to argparse.RawDescriptionHelpFormatter, but
    retains line breaks when formatting the help message."""

    def _fill_text(self, text, width, indent=""):
        lines = text.splitlines()
        a = [
            textwrap.fill(line, width, initial_indent=indent, subsequent_indent=indent)
            for line in lines
        ]
        return "\n".join(a)

    def _split_lines(self, text, width):
        return self._fill_text(text, width).split("\n")


class Option(Enum):
    """Common recipe options"""

    OUTPUT = 0
    FORCE_OVERWRITE = 1
    ROWS = 2
    START = 3
    END = 4
    NVTX = 5
    BASE = 6
    MANGLED = 7
    BINS = 8
    CSV = 9
    INPUT = 10
    DISABLE_ALIGNMENT = 11
    SUBTIMERANGE = 12


class ModeAction(argparse.Action):
    def __init__(self, **kwargs):
        kwargs.setdefault(
            "choices",
            tuple(mode.name.replace("_", "-").lower() for mode in recipe.Mode),
        )
        super().__init__(**kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        value = recipe.Mode[values.replace("-", "_").upper()]
        setattr(namespace, self.dest, value)


class InputAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        # Remove any inner lists.
        flattened_list = []
        for value in values:
            if isinstance(value, list):
                flattened_list.extend(value)
            else:
                flattened_list.append(value)
        setattr(namespace, self.dest, flattened_list)


class ArgumentParser(argparse.ArgumentParser):
    """Custom argument parser with predefined arguments"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._context_group = self.add_argument_group("Context")
        self._recipe_group = self.add_argument_group("Recipe")

    @property
    def recipe_group(self):
        return self._recipe_group

    def add_recipe_argument(self, option, *args, **kwargs):
        self.add_argument_to_group(self._recipe_group, option, *args, **kwargs)

    def add_argument_to_group(self, group, option, *args, **kwargs):
        if not isinstance(option, Option):
            group.add_argument(option, *args, **kwargs)
            return

        if option == Option.OUTPUT:
            group.add_argument(
                "--output",
                type=process_output,
                help="Output directory name.\n"
                "Any %%q{ENV_VAR} pattern in the filename will be substituted with the value of the environment variable.\n"
                "Any %%h pattern in the filename will be substituted with the hostname of the system.\n"
                "Any %%p pattern in the filename will be substituted with the PID.\n"
                "Any %%n pattern in the filename will be substituted with the minimal positive integer that is not already occupied.\n"
                "Any %%%% pattern in the filename will be substituted with %%",
                **kwargs,
            )
        elif option == Option.FORCE_OVERWRITE:
            group.add_argument(
                "--force-overwrite",
                action="store_true",
                help="Overwrite existing directory",
                **kwargs,
            )
        elif option == Option.ROWS:
            group.add_argument(
                "--rows",
                metavar="limit",
                type=int,
                default=-1,
                help="Maximum number of rows per input file",
                **kwargs,
            )
        elif option == Option.START:
            group.add_argument(
                "--start",
                metavar="time",
                type=int,
                help="Start time used for filtering in nanoseconds",
                **kwargs,
            )
        elif option == Option.END:
            group.add_argument(
                "--end",
                metavar="time",
                type=int,
                help="End time used for filtering in nanoseconds",
                **kwargs,
            )
        elif option == Option.NVTX:
            group.add_argument(
                "--nvtx",
                metavar="range[@domain]",
                type=str,
                help="NVTX range and domain used for filtering",
                **kwargs,
            )
        elif option == Option.BASE:
            group.add_argument(
                "--base", action="store_true", help="Kernel base name", **kwargs
            )
        elif option == Option.MANGLED:
            group.add_argument(
                "--mangled", action="store_true", help="Kernel mangled name", **kwargs
            )
        elif option == Option.BINS:
            group.add_argument(
                "--bins",
                type=process_integer(0),
                default=30,
                help="Number of bins",
                **kwargs,
            )
        elif option == Option.CSV:
            group.add_argument(
                "--csv",
                action="store_true",
                help="Additionally output data as CSV",
                **kwargs,
            )
        elif option == Option.INPUT:
            group.add_argument(
                "--input",
                type=process_input,
                default=None,
                nargs="+",
                action=InputAction,
                help="One or more paths to nsys-rep files or directories.\n"
                "Directories can optionally be followed by ':n' to limit the number of files",
                **kwargs,
            )
        elif option == Option.DISABLE_ALIGNMENT:
            group.add_argument(
                "--disable-alignment",
                action="store_true",
                help="Disable automatic session alignment.\n"
                "By default, session times are aligned based on the epoch time of the report file collection.\n"
                "This option will instead use relative time, which is useful for comparing individual sessions",
                **kwargs,
            )
        elif option == Option.SUBTIMERANGE:
            group.add_argument(
                "--subtimerange",
                metavar="subrange",
                type=range_type,
                help="time range used for filtering in nanoseconds, --subtimerange 10.452e9-10.587e9,10.601e9-10.882e9, \n"
                "it means you want to stat the time range from 10.452s to 10.587s , and also 10.601s to 10.882s",
                **kwargs,
            )
        else:
            raise NotImplementedError

    def add_context_arguments(self):
        self._context_group.add_argument(
            "--mode",
            action=ModeAction,
            default=recipe.Mode.CONCURRENT,
            help="Mode to run tasks",
        )
