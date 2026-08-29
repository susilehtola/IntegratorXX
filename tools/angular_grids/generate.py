"""Regenerate an angular grid table at arbitrary precision.

    python3 -m angular_grids.generate --family lebedev_laikov --npts 110 \
        --digits 40 --out gen/

Reads the tabulated double-precision rule, refines it by Gauss-Newton on the
exactness conditions, verifies the result against the full redundant set of
monomial conditions, and writes a header carrying the digits as strings.
"""
import argparse
import sys
from pathlib import Path

from mpmath import mp

from .emit import emit
from .refine import decompose, materialise, refine, verify
from .tables import grid_path, read_grid


def algebraic_order(family, npts, root):
    """Read the order off the family's dispatch header."""
    import re
    text = (Path(root) / "include/integratorxx/quadratures/s2" / f"{family}.hpp").read_text()
    body = text[text.index("algebraic_order_by_npts"):]
    m = re.search(r"case\s+%d\s*:\s*\n?\s*return\s+(\d+)" % npts, body)
    if not m:
        raise SystemExit(f"no algebraic order tabulated for {family} with {npts} points")
    return int(m.group(1))


OCTAHEDRAL = ("lebedev_laikov", "delley")


def _run_octahedral(family, npts, order, digits, root, verbose):
    points, weights = read_grid(grid_path(root, family, npts))
    scale = 4 * mp.pi / sum(weights)          # these tables are normalised to 1
    orbits = decompose(points, [w * scale for w in weights])
    n_param = sum(1 + o.type.n_param for o in orbits)
    if verbose:
        print(f"{family}_{npts}: order {order}, {len(orbits)} octahedral orbits, "
              f"{n_param} free parameters")
    refine(orbits, order, digits + 5, verbose=verbose)
    P, W = materialise(orbits)
    return P, W, verify(P, W, order)


def _run_icosahedral(family, npts, order, digits, root, verbose):
    from . import ab_refine as ab
    points, weights = read_grid(grid_path(root, family, npts))
    orbits = ab.decompose(points, weights)    # AB tables already sum to 4 pi
    if verbose:
        print(f"{family}_{npts}: order {order}, {len(orbits)} icosahedral orbits, "
              f"{ab.n_param(orbits)} free parameters")
    conds = ab.select_conditions(orbits, order, verbose=verbose)
    ab.refine(orbits, order, digits + 5, conds=conds, verbose=verbose)
    P, W = ab.materialise(orbits)
    return P, W, ab.verify(orbits, order)


def _run_design(family, npts, order, digits, root, verbose):
    from . import designs
    points, _ = read_grid(grid_path(root, family, npts))
    d = designs.Design(points, order)
    if verbose:
        print(f"{family}_{npts}: order {order}, equal-weight design, "
              f"{2 * npts} tangential unknowns")
    designs.refine(d, digits + 5, verbose=verbose)
    return d.points, [d.weight] * d.n, designs.verify(d)


def run(family, npts, digits, root, out, verbose=True):
    mp.dps = digits + 20                      # guard digits for the refinement
    order = algebraic_order(family, npts, root)
    if family in OCTAHEDRAL:
        P, W, err = _run_octahedral(family, npts, order, digits, root, verbose)
    elif family == "ahrens_beylkin":
        P, W, err = _run_icosahedral(family, npts, order, digits, root, verbose)
    elif family == "womersley":
        P, W, err = _run_design(family, npts, order, digits, root, verbose)
    else:
        raise SystemExit(f"unknown family {family}")

    if len(P) != npts:
        raise SystemExit(f"materialised {len(P)} points, expected {npts}")
    if err > mp.mpf(10) ** (-digits):
        raise SystemExit(f"refinement reached only {mp.nstr(err, 4)}, short of {digits} digits")
    if verbose:
        print(f"  verified worst error {mp.nstr(err, 4)} against the full "
              f"condition set for order {order}")

    text = emit(family, npts, order, P, W, digits, err)
    if out:
        out = Path(out)
        # Writing into the source directory destroys the input: a partial run
        # replaces the literal tables the remaining sizes still need to read,
        # and every later size then fails to parse. Generate elsewhere and
        # install deliberately.
        src = (Path(root) / "include/integratorxx/quadratures/s2" / family).resolve()
        if out.resolve() == src:
            raise SystemExit(
                f"refusing to write into the source directory {src}; "
                "generate into a separate directory and copy the results in")
        path = out / f"{family}_{npts}.hpp"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        if verbose:
            print(f"  wrote {path}")
    return text, err


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--family", required=True,
                    choices=["lebedev_laikov", "delley", "ahrens_beylkin", "womersley"])
    ap.add_argument("--npts", required=True, type=int)
    ap.add_argument("--digits", type=int, default=40)
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    run(a.family, a.npts, a.digits, a.root, a.out)


if __name__ == "__main__":
    sys.exit(main())
