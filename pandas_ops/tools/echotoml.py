from pathlib import Path
from pprint import pprint

import click
import toml


@click.command(context_settings={"show_default": True})
@click.argument("toml_file", type=Path)
def echotoml(toml_file: Path) -> None:
    with open(toml_file, "r") as fh:
        conf = toml.load(fh)
    pprint(conf)
