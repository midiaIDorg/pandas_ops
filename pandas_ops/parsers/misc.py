def parse_key_equal_value(keyval: str) -> tuple:
    k, v = keyval.split("=")
    try:
        v = int(v)
    except ValueError:
        try:
            v = float(v)
        except ValueError:
            pass
    return (k, v)
