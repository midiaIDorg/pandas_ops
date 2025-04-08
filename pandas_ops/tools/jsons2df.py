import argparse
import json
import pandas as pd

from pandas_ops.io import save_df
from pathlib import Path


def load_json_files(paths):
    """ChatGPT generated code."""
    records = []

    for path_str in paths:
        path = Path(path_str).resolve()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {path}: {e}")
            continue

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    item["_source_path"] = str(path)
                    records.append(item)
                else:
                    print(f"Skipping non-dict item in list from {path}")
        elif isinstance(data, dict):
            data["_source_path"] = str(path)
            records.append(data)
        else:
            print(f"Unsupported JSON structure in {path}")

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(
        description="Load multiple JSON files into a DataFrame"
    )
    parser.add_argument("output", help="Path to output table.")
    parser.add_argument("paths", nargs="+", help="Paths to JSON files")
    args = parser.parse_args()

    df = load_json_files(args.paths)
    print(df)

    save_df(df, args.output)


if __name__ == "__main__":
    main()
