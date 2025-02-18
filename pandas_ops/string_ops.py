import re

from itertools import product

from pathlib import Path

import typing


def iter_glob(
    path: str | Path,
    _brackets_pattern: re.Pattern = re.compile(r"\{(.*?)\}"),
    _sub_bracket_sign: str = "{}",
    _split_sign: str = ",",
) -> typing.Iterator[Path]:
    bracket_occurences = [
        o.split(_split_sign) for o in _brackets_pattern.findall(str(path))
    ]
    clean_text = _brackets_pattern.sub(_sub_bracket_sign, str(path))
    for fills in product(*bracket_occurences):
        yield Path(clean_text.format(*fills))


def test_iter_glob():
    path = "stats/**/*.{csv,json}_{1,2,4}"
    expected_outcome = [
        "stats/**/*.csv_1",
        "stats/**/*.csv_2",
        "stats/**/*.csv_4",
        "stats/**/*.json_1",
        "stats/**/*.json_2",
        "stats/**/*.json_4",
    ]
    assert list(map(str, iter_glob(path))) == expected_outcome
