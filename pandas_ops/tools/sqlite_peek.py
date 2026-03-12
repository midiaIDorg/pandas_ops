import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Dump the first K rows of every table in a SQLite database to a text file for AI-readable overview."
    )
    parser.add_argument(
        "db_path",
        type=Path,
        help="Path to the SQLite database file.",
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help="Path to the output text file. Use '-' to write to stdout.",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=10,
        help="Number of rows to show per table (default: 10).",
    )
    args = parser.parse_args()

    con = sqlite3.connect(args.db_path)
    tables = [
        row[0]
        for row in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    ]

    out = sys.stdout if str(args.output_path) == "-" else open(args.output_path, "w")
    try:
        for table in tables:
            total = con.execute(f"SELECT COUNT(*) FROM [{table}]").fetchone()[0]
            df = pd.read_sql(f"SELECT * FROM [{table}] LIMIT {args.n}", con)
            out.write(f"=== Table: {table} ({total} rows total) ===\n")
            out.write(df.to_string(index=False))
            out.write("\n\n")
    finally:
        if out is not sys.stdout:
            out.close()
    con.close()


if __name__ == "__main__":
    main()
