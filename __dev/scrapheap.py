%load_ext autoreload
%autoreload 2
import h5py
import matplotlib
import matplotlib.pyplot as plt
import MSclusterparser.raw_peaks_4DFF_parser as parsers
import numpy as np
import pandas as pd
from MSclusterparser.hdf_ops import create_empty_hdf
from pandas_ops.io import read_data_pd
from tqdm import tqdm


def hdf2dict(
    path,
    group: str="raw/data",
    start: int=0,
    stop:  int=-1,
    columns: None | list[str] = None,
) -> pd.DataFrame:
    with h5py.File(path, "r") as f:
        _cols = set(f['raw/data'])
        if columns is None:
            columns = _cols 
        else:
            for c in columns:
                assert c in _cols, f"Missing column {c}"
        return {p: f[f'raw/data/{p}'][start:stop] for p in columns}
            

pd.DataFrame(hdf2dict("clusters/8027/i4DFF/clusters.hdf", stop=10, columns=["intensity","ClusterID"]))



clusters_cnts = []
for hpr_path in tqdm(["clusters/8027_THPR67_d4/i4DFF/clusters.hdf"]):
    clusters = parsers.Clusters_4DFF_HDF(str(hpr_path))
    clusters_cnts.append(len(clusters))

total_clusters = sum(clusters_cnts)
plt.plot(clusters_cnts)
plt.title(f"{wildcards.dataset}, THPRs, diagonals={int(wildcards.diagonals_cnt)}, total clusters = {total_clusters:_}")
