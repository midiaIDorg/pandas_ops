#!/usr/bin/env python3
""" 

Potential generalization:
    * table paths passed in as a map name->path
    * this would enable operations on multiple tables.

"""
from pathlib import Path
from pprint import pprint

import click
import duckdb

import tomllib


class KeyValueAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        super(KeyValueAction, self).__init__(option_strings, dest, nargs, **kwargs)
        self.kv_dict = {}

    def __call__(self, parser, namespace, values, option_string=None):
        pprint(values)
        for value in values:
            key, value = value.split("=", maxsplit=1)
            self.kv_dict[key] = value
        setattr(namespace, self.dest, self.kv_dict)


def main():
    parser = argparse.ArgumentParser(description="Perform SQL on tables.")
    parser.add_argument(
        "--named_path",
        action=KeyValueAction,
        nargs="+",
        required=True,
        metavar="KEY=VALUE",
    )
    parser.add_argument(
        "--config",
        help="Path to config (toml format) defining under `filter` the SQL expression used to read, filter, and save the table in the format of interest.",
        default=None,
        type=Path,
    )
    parser.add_argument(
        "--sql",
        help="Directly execute the provided sql string.",
        default=None,
        type=str,
    )
    parser.add_argument("--verbose", action="store_true", help="Be more verbose")
    args = parser.parse_args()
    assert (args.config is None) ^ (
        args.sql is None
    ), "Provide EITHER an sql or a path to the config toml, not both, not none."
    if not args.config is None:
        with open(args.config, "rb") as h:
            config = tomllib.load(h)
        sql = config["filter"].format(**args.named_path)
    else:
        sql = args.sql.format(**args.named_path)

    if args.verbose:
        from pprint import pprint

        print("Filtering table:")
        pprint(sql)
    duckdb.query(sql)


if __name__ == "__main__":
    main()
