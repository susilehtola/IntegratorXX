# Angular grid generation

Regenerates the Lebedev-Laikov and Delley tables at arbitrary precision.

## Why

The tabulated grids carry about 16 significant digits, so `LebedevLaikov<T>`
cannot be more accurate than `double` however wide `T` is. Worse, the tables
are written as unsuffixed floating literals inside `std::array<T,N>`, and an
unsuffixed literal has type `double` whatever `T` is -- so simply writing more
digits into the existing format would change nothing.

Measured on the 110-point rule, integrating `x^4 y^2 z^2`:

| precision          | current table    | regenerated table |
|--------------------|------------------|-------------------|
| `double`           | 1.91e-15         | 1.74e-15          |
| `long double`      | 1.26e-15         | **2.55e-19**      |
| `cpp_bin_float_50` | not achievable   | **1.94e-40**      |

## How

Finding a spherical quadrature rule and refining one are very different
problems, and only the second is needed here:

* **Construction** is a global optimisation with many local minima. It needs
  second-order globalisation -- this is what a trust-region solver such as
  [OpenTrustRegion](https://github.com/eriksen-lab/opentrustregion) is for --
  but `double` precision suffices, since the goal is only to land in the right
  basin.
* **Refinement** starts from a rule that is already correct to 16 digits and is
  purely local. Newton converges quadratically, so two or three iterations
  reach any precision asked for, with no globalisation at all.

For the existing families construction is already done: the tables *are* the
result. Only refinement is needed, and that is what this tool does.

Tractability comes from symmetry. Both families are invariant under the full
octahedral group, so points fall into orbits carrying one weight and zero, one
or two angular parameters:

| grid              | points | orbits | free parameters |
|-------------------|-------:|-------:|----------------:|
| Lebedev-Laikov 50 |     50 |      4 |               5 |
| Lebedev-Laikov110 |    110 |      6 |              10 |
| Lebedev-Laikov590 |    590 |     20 |              44 |
| Lebedev-Laikov5810|   5810 |    144 |             385 |

So the largest Lebedev rule is a 385-parameter dense Newton solve, not a
17430-parameter one.

The exactness conditions are imposed on even monomials. On the unit sphere
`z^2 = 1 - x^2 - y^2`, so every even monomial reduces to a fixed linear
combination of `x^(2a) y^(2b)`; imposing only those implies the rest, at about
a seventh of the cost. Results are always *verified* against the full redundant
set.

## Usage

```sh
PYTHONPATH=tools python3 -m angular_grids.generate \
    --family lebedev_laikov --npts 110 --digits 40 --out gen/
```

Requires `mpmath`. Output headers store the digits as decimal strings and
convert them through `detail::grid_scalar<T>::parse` at grid construction,
which works for the built-in types as well as Boost.Multiprecision and MPFR.

## Not covered

* **Ahrens-Beylkin** has icosahedral rather than octahedral symmetry (15012
  points / 60 = 250.2 against 251 distinct weights, the rotation group without
  inversion). Same method, different orbit algebra.
* **Womersley** grids are equal-weight spherical designs with no symmetry at
  all: 2N angular unknowns, and the Jacobian is rank-deficient by exactly three
  because any rotation of a design is a design. They refine fine, but need a
  matrix-free least-squares step (CGLS) rather than a dense solve, and at 7939
  points the residual evaluation wants a compiled high-precision kernel.
* **Constructing new rules**, including regenerating the corrupt 552-point
  Ahrens-Beylkin table. That is the trust-region half of the problem.
