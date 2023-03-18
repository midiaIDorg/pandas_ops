import pandas as pd


def get_to_show(data, row_count, pandas_style, columns_only):
    if columns_only:
        return data.columns
    if pandas_style:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_rows", row_count)
        return data
    return data.head(row_count).to_csv(index=data.index.name == "")
