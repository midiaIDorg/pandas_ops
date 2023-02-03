import os


def read_data_pd(file_path, **kwargs):
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    file_readers = {
        ".csv": pd.read_csv,
        ".tsv": lambda x: pd.read_csv(x, sep="\t"),
        ".xlsx": pd.read_excel,
        ".json": pd.read_json,
        ".feather": pd.read_feather,
        ".hdf": pd.read_hdf,
    }

    reader = file_readers.get(file_extension)
    if reader is None:
        raise ValueError(f"Unsupported file extension: {file_extension}")

    return reader(file_path, **kwargs)
