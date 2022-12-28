import replicator
import numpy as np
import sympy as sym
import fractions

payoffs_m = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])


def test_odes():
    xs = np.array([1 / 3, 1 / 3, 1 / 3])
    assert np.allclose(
        replicator.odes(xs, None, payoffs_m), np.array([0, 0, 0])
    )


def test_fixed_points():
    size = payoffs_m.shape[0]
    xs = np.array(sym.symbols(f"x_1:{size + 1}"))

    solutions = replicator.fixed_points(xs, payoffs_m)

    assert len(solutions) == 4
    assert isinstance(solutions, list)
    assert (0, 0, 1) in solutions
    assert (0, 1, 0) in solutions
    assert (1, 0, 0) in solutions
    assert (
        fractions.Fraction(1, 3),
        fractions.Fraction(1, 3),
        fractions.Fraction(1, 3),
    ) in solutions


def test_jacobian():
    size = payoffs_m.shape[0]
    xs = np.array(sym.symbols(f"x_1:{size + 1}"))
    J = replicator.jacobian(xs, payoffs_m)
    expected = sym.Matrix(
        [
            [-(xs[1] - xs[2]), -xs[0], xs[0]],
            [xs[1], xs[0] - xs[2], -xs[1]],
            [-xs[2], xs[2], -(xs[0] - xs[1])],
        ]
    )
    assert J == expected


def tests_point_is():
    size = payoffs_m.shape[0]
    xs = np.array(sym.symbols(f"x_1:{size + 1}"))
    J = replicator.jacobian(xs, payoffs_m)

    for point in [
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1]),
    ]:
        is_ = replicator.point_is(J, xs, point)

        assert is_ == "saddle"

    mixed = replicator.point_is(
        J,
        xs,
        np.array(
            [
                fractions.Fraction(1, 3),
                fractions.Fraction(1, 3),
                fractions.Fraction(1, 3),
            ]
        ),
    )
    assert mixed == "saddle"
