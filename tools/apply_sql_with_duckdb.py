#!/usr/bin/env python3
""" 

Potential generalization:
    * table paths passed in as a map name->path
    * this would enable operations on multiple tables.

"""
import argparse
import json
from pathlib import Path
from pprint import pprint

import duckdb
import tomllib
from pandas_ops.misc import in_ipython


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


if in_ipython():
    from IPython import get_ipython

    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

    class args:
        named_paths = dict(
            source="partial/G8027/G8045/MS1@tims@08f35c4405b@default@fast@default@MS2@tims@29f95c3131e@default@fast@default/matcher@prtree@narrow/rough@default/1stSearch@sage@95c2993@p12f15nd/fasta@3/results.sage.tsv",
            target="/tmp/filtered.sage.results.parquet",
        )
        config = "configs/search/output_filters/sage/default.toml"
        sql = None
        verbose = True

    class args:
        named_paths = dict(
            source="outputs/doubleSageWithCheese/G8027/G8045/sage/second_gen/mz_recalibration_edge_refinement/precursors_without_multiply_assigned_fragments.parquet",
            target="/tmp/stripped_sequences.csv",
        )
        config = "configs/search/output_filters/sage/strip_sequences.toml"
        sql = None
        verbose = True

else:
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
    from pprint import pprint

    pprint(args.__dict__)


if __name__ == "__main__":
    assert (args.config is None) ^ (
        args.sql is None
    ), "Provide EITHER an sql or a path to the config toml, not both, not none."
    if not args.config is None:
        with open(args.config, "rb") as h:
            config = tomllib.load(h)
        sql = config["filter"].format(**args.named_path)
    else:
        sql = args.sql.format(source=args.source, target=args.target)

    if args.verbose:
        from pprint import pprint

        print("Filtering table:")
        pprint(sql)
    duckdb.query(sql)
