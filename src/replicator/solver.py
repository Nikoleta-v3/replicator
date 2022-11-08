"""Files contains all the functions necessary for the odes,
points and stability calculations."""

import sympy as sym

import numpy as np


def odes(x, time, payoffs):
    ax = np.dot(payoffs, x)
    return x * (ax - np.dot(x, ax))


def fixed_points(xs, payoffs):
    """Solves symbolically the system of equations of the replicator dynamics.

    Parameters
    ----------
    xs : list
        A list of sympy.Symbols
    payoffs : np.array
        The payoffs matrix

    Returns
    -------
    list
        The fixed points
    """

    x_bar = sym.solve([sum(xs) - 1] + list(odes(xs, None, payoffs)), list(xs))
    return x_bar


def jacobian(xs, payoff):
    return sym.Matrix(
        [
            [sym.diff(expression, x).factor() for x in xs]
            for expression in odes(xs, None, payoff)
        ]
    )


def point_is(jacobian, xs, solution):

    eigen_values = jacobian.subs(
        {x: s for x, s in zip(xs, solution)}
    ).eigenvals()
    eigen_values = np.array(
        [float(val.subs({sym.I: 0})) for val in list(eigen_values.keys())]
    )
    if 1 in solution:
        mask = np.ones(eigen_values.shape, bool)
        mask[np.where(np.isclose(solution, 1))] = False
        eigen_values = eigen_values[mask]

    if (eigen_values < 0).all():
        return "sink"
    if (eigen_values > 0).all():
        return "source"
    else:
        return "saddle"
