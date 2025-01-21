%load_ext autoreload
%autoreload 2
import multiprocessing as mp
from pathlib import Path
from subprocess import run

import click
from tqdm import tqdm

import pandas as pd
from kilograms import scatterplot_matrix
from MSclusterparser.raw_peaks_4DFF_parser import Clusters_4DFF_HDF
from pandas_ops.io import read_df, save_df

## Ye Old Pipeline.
# ./run snakemake -call outputs/all/G8027/G8045


def _run(source_target):
    source, target = source_target
    run(
        [
            "run_general_sql",
            "SUMMARIZE (SELECT * FROM '{source}');",
            "--source",
            source,
            "--target",
            target,
        ],
        check=True,
    )


old_pipeline_output_folder = Path(
    "/home/matteo/Projects/midia/docker_images/midia_docker/dockerhub/outputs/debug/G8027/None"
)
new_pipeline_output_folder = Path(
    "/home/matteo/Projects/midia/midia_experiments/matteo_devel/pipelines/devel_no_ms2rescore2/midia_pipe/outputs/base/debug/G8027/None"
)
new_pipeline_output_stats = list(
    (new_pipeline_output_folder / "stats/tables/").glob("*")
)
assert old_pipeline_output_folder.exists()

num_processes = 16
output_folder = old_pipeline_output_folder / "table_summaries"
output_folder.mkdir(exist_ok=True)

source_targets = [
    (str(source), str(output_folder / f"{source.name}.summary.csv"))
    for source in old_pipeline_output_folder.glob("tables/*")
]

with mp.Pool(processes=num_processes) as pool:
    results = pool.map(_run, source_targets)


## Getting towards kilograms.
# old_edges, old_MS1_stats, old_MS2_stats = (
#     read_df(old_pipeline_output_folder / "tables" / table)
#     for table in [
#         "refined_edges.startrek",
#         "refined_precursor_stats.parquet",
#         "refined_fragment_stats.parquet",
#     ]
# )


def append_cols(df, **kwargs):
    for arg, val in kwargs.items():
        df[arg] = val
    return df


def get_stats(paths, file_extractor=lambda x: x.stem):
    return pd.concat((append_cols(read_df(x), file=file_extractor(x)) for x in paths))


def map_replace(x, *replacements: tuple[str, str]):
    res = x
    for to_replace, replacement in replacements:
        res = res.replace(to_replace, replacement)
    return res


new_pipe = append_cols(
    get_stats(
        new_pipeline_output_stats,
        file_extractor=lambda x: map_replace(
            x.stem, ("_summary", ""), (".startrek.summary", "")
        ),
    ),
    pipeline="new",
)
old_pipe = append_cols(
    get_stats(
        (old_pipeline_output_folder / "table_summaries").glob("*"),
        file_extractor=lambda x: map_replace(
            x.stem, (".parquet.summary", ""), (".startrek.summary", "")
        ),
    ),
    pipeline="old",
)

_all = pd.concat([new_pipe, old_pipe])

simple_summary = pd.DataFrame(
    [
        {"pipeline": pipe, "column": col, "cnt": int(df["count"].iloc[0])}
        for (pipe, col), df in _all.groupby(["pipeline", "file"])
    ]
)

simple_summary[["column", "pipeline", "cnt"]].sort_values(
    ["column", "pipeline"], ignore_index=True
)

newMS1clusters = Clusters_4DFF_HDF(new_pipeline_output_folder / "raw/clusters/ms1.hdf")
oldMS1clusters = Clusters_4DFF_HDF(old_pipeline_output_folder / "clusters/MS1.hdf")

newMS2clusters = Clusters_4DFF_HDF(new_pipeline_output_folder / "raw/clusters/ms2.hdf")
oldMS2clusters = Clusters_4DFF_HDF(old_pipeline_output_folder / "clusters/MS2.hdf")

for newClust, oldClust in tqdm(zip(newMS2clusters, oldMS2clusters), total=len(newMS2clusters)):
    if len(oldClust.compare(newClust)) != 0:
        print(oldClust)
        print(newClust)
        print(oldClust.compare(newClust))
        break


oldClust = oldMS2clusters.query(i)
newClust = newMS2clusters.query(i)
oldClust.compare(newClust)

# so indeed it seems that tims clusters are fully reproducible (intriguing, would have expected some differences on a multithreaded pc)

old_precursor_cluster_stats = read_df(old_pipeline_output_folder / "tables/precursor_cluster_stats.parquet")
new_precursor_cluster_stats = read_df(new_pipeline_output_folder / "stats/clusters/ms1.parquet")


def VennIt(A, B):
    sA, sB = map(set, (A, B) )
    return map(list, (sA & sB, sA - sB, sB - sA))

intesection_cols, _, _ = VennIt(
    old_precursor_cluster_stats.columns,
    new_precursor_cluster_stats.columns
)

oldstats = old_precursor_cluster_stats[intesection_cols]
newstats = new_precursor_cluster_stats[intesection_cols]


(oldstats - newstats).abs().max()
