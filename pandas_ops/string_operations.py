import re


class PatternReplacer:
    """
    Match a sequence of predetermined mappable patterns and do replacements.

    Arguments:
        replacements (dict): A dictionary specifying which stringe to replace with what.
        pattern (str): A general pattern fitting all of replacement keys.
    """

    def __init__(
        self,
        replacements: dict[str, str],
        pattern: str | re.Pattern = r"\[.*?\]",
    ):
        self.pattern = re.compile(pattern)
        self.replacements = replacements
        for _in, _out in replacements.items():
            assert (
                len(re.findall(self.pattern, _in)) > 0
            ), f"The submitted replacemnt, `{_in}->{_out}`, cannot be used with pattern `{pattern}`."

    def apply(self, string: str) -> str:
        """Apply a replacement.

        Arguments:
            string (str): a string to be searched with pattern and to whom the replacements will apply.

        Returns:
            str: The string with all replacements.
        """
        out_sequence = string
        for _in in set(re.findall(self.pattern, string)):
            try:
                _out = self.replacements[_in]
            except KeyError:
                raise KeyError(
                    f"Modification {_in} not among those specified in the replacements."
                )
            out_sequence = out_sequence.replace(_in, _out)
        return out_sequence
