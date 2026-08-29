"""Read the tabulated angular grids out of the generated C++ headers."""
import re
from pathlib import Path

from mpmath import mpf

_NUM = re.compile(r"[-+]?\d*\.\d+[EeDd][-+]?\d+")


def _numbers(text):
    return [mpf(t.replace("D", "e").replace("E", "e")) for t in _NUM.findall(text)]


def _section(text, name):
    i = text.index(name + " = {")
    j = text.index("};", i)
    return text[i:j]


def read_grid(path):
    """Return (points, weights) from a *_<npts>.hpp data header."""
    text = Path(path).read_text()
    flat = _numbers(_section(text, "points"))
    weights = _numbers(_section(text, "weights"))
    points = [tuple(flat[3 * i:3 * i + 3]) for i in range(len(flat) // 3)]
    if len(points) != len(weights):
        raise ValueError(f"{path}: {len(points)} points but {len(weights)} weights")
    return points, weights


def grid_path(root, family, npts):
    return Path(root) / "include/integratorxx/quadratures/s2" / family / f"{family}_{npts}.hpp"
