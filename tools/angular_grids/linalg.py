"""Small dense solves shared by the refinement paths.

The Gauss-Newton systems here are tiny but not always full rank, and mpmath's
lu_solve and qr_solve both refuse a singular matrix outright rather than
returning something usable. Working precision changes whether a given system
looks singular -- delley_1454 solves cleanly at 40 digits and fails at 20 --
so this is a robustness property rather than a property of any one grid.
"""
from mpmath import lu_solve, matrix, mp, mpf, sqrt


def independent_columns(J, tol=mpf("1e-16")):
    """Indices of a maximal independent set of columns, by modified Gram-Schmidt.

    The threshold is relative to each column's own norm, so columns that differ
    wildly in scale (a weight column against an angular one) are judged fairly.
    """
    keep, basis = [], []
    for j in range(J.cols):
        v = [J[i, j] for i in range(J.rows)]
        nrm0 = sqrt(mp.fsum(x * x for x in v))
        if nrm0 == 0:
            continue
        for u in basis:
            d = mp.fsum(v[i] * u[i] for i in range(len(v)))
            v = [v[i] - d * u[i] for i in range(len(v))]
        nrm = sqrt(mp.fsum(x * x for x in v))
        if nrm > tol * nrm0:
            basis.append([x / nrm for x in v])
            keep.append(j)
    return keep


def lstsq_step(J, F):
    """Gauss-Newton step, solved only in the directions the Jacobian resolves.

    Directions outside the resolved set are left at zero: the conditions cannot
    see them, so moving along them is guesswork.
    """
    keep = independent_columns(J)
    m = len(keep)
    if m == 0:
        return [mpf(0)] * J.cols
    JtJ = matrix(m, m)
    Jtf = matrix(m, 1)
    for a in range(m):
        ja = keep[a]
        for b in range(a, m):
            jb = keep[b]
            v = mp.fsum(J[k, ja] * J[k, jb] for k in range(J.rows))
            JtJ[a, b] = v
            JtJ[b, a] = v
        Jtf[a] = mp.fsum(J[k, ja] * F[k] for k in range(J.rows))
    red = lu_solve(JtJ, -Jtf)
    step = [mpf(0)] * J.cols
    for a, j in enumerate(keep):
        step[j] = red[a]
    return step
