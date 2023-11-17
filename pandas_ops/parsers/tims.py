from pathlib import Path


def parse_value(value: str) -> str | float | int:
    try:
        parsed_value = float(value)
        if not "." in value:
            parsed_value = int(parsed_value)
        return parsed_value
    except ValueError:
        return value


def read_tims_config(path):
    res = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line != "" and line[0] != "#":
                parameter, value = line.split("=")
                assert (
                    parameter not in res
                ), f"Found {parameter} with two values: {res[parameter]} and {value}."
                res[parameter] = parse_value(value)
    return res


def write_tims_config(tims_config: dict[str, str], out: Path | str | None = None):
    text = "\n".join((f"{k} = {v}" for k, v in tims_config.items())) + "\n"
    if out is None:
        print(text)
    else:
        with open(out, "w") as f:
            f.write(text)
