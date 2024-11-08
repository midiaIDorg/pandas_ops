#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd
import tomllib
from pandas_ops.io import read_df

# class args:  # quick mock
#     source = Path(
#         "P/cluster_stats/raw/G4ACAESDtcV2e3EgU-kvIjds8KboFIf8Xn3Qa309X7yBeZxhuk00qK9TZY52RmTivlOFiMa5sqTiXgRxmz9fAR5dmeWrWoFzG8jxT-NXrVCxP9uULLUlYj3amleJZ2BYvjQprvIrTQrglXHrwXFVmF1j-21ouqEFhx1Hd-7rkj4bCCGfWlhxvNo5BeqB_0fCpsuBHBcLuz42Y1nLsE4ieTmQVLs9uPg11ifi9XKopKHp_xiq890yBQ==/fragment_stats.parquet",
#     )
#     # source = Path(
#     #     "P/edges/G68GIORwbbHtMM1U-tTfZOqmexTy_25M4HQsXBO0C5dGnFKK9ZTNNW0tNt4myoGwXtbSsNnuCGi92Hj4MTJEm1C3U5lsI4ehXXkq5tHU1kLxH1JR5UHpwyBXmKOp1-/g2zuBbOv9ARv7r5DjxOh6PBoaQqBp7hCaw5N5oJ8BjZ_U0SPg9Dqc7MlT9IjKUgA2Pp1hHODpzvtvn8fo8wcFOgRb1H15rAoT0UpHjTxtspj5oFkTTzb5QxeaVSgIbvJIlvZZDP9uJaqz3DRdUdCtP-ylx-u-pUDTv4544DXhl6KrJuCGyyxU-i6LRRKlRoLF8TOU-FLJU8weYKqyz2HKxOZzEtRrtmAE=/rough.startrek"
#     # )
#     target = "/tmp/testing_stats_table.csv"
#     config = "configs/table_stats/table_summary.toml"


# pd.set_option("display.max_columns", None)
# pd.set_option("display.max_rows", 5)

parser = argparse.ArgumentParser(
    description="Summarize a table.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("source", help="Path to the table to summarize.", type=Path)
parser.add_argument("config", help="Path to the table to summarize.", type=Path)
parser.add_argument("target", help="Path to the summary table.", type=Path)
args = parser.parse_args()


def main(args):
    with open(args.config, "rb") as f:
        config = tomllib.load(f)
    duckcon = duckdb.connect()

    source = args.source
    if source.suffix == ".startrek":
        df = read_df(source)
        source = "df"
    _query = config["sql"].format(source=source, target=args.target)
    duckcon.execute(_query)


if __name__ == "__main__":
    main(args)
