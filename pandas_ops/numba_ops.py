import functools


def inputs_series_to_numpy(foo):
    @functools.wraps(foo)
    def wrapper(*args, **kwargs):
        args = [arg.to_numpy() if isinstance(arg, pd.Series) else arg for arg in args]
        kwargs = {
            k: np.array(v) if isinstance(v, pd.Series) else v for k, v in kwargs.items()
        }
        return foo(*args, **kwargs)

    return wrapper
