"""Files contains all the functions necessary for the odes,
points and stability calculations."""

import sympy as sym

import numpy as np

from scipy.optimize import fsolve


def odes(x, time, payoffs, mutation=None):
    if mutation is None:
        mutation = 0
    f = payoffs @ x
    phi = x.T @ f
    return x * (f - phi - 3 * mutation) + mutation


def odes_for_numerical_solver(p, payoff_mat, mutation):
    size = payoff_mat.shape[0]
    xs = np.array(sym.symbols(f"x_1:{size + 1}"))
    odes_ = odes(xs, None, payoff_mat, mutation)

    fs = [sym.lambdify(xs, d) for d in odes_]

    return [f(*p) for f in fs]


def fixed_points(xs, payoffs, mutation=None, starting_solution=None):
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
    if mutation:
        x_bar = fsolve(
            odes_for_numerical_solver,
            starting_solution,
            args=(
                payoffs,
                mutation,
            ),
        )
    else:
        x_bar = sym.solve(
            [sum(xs) - 1] + list(odes(xs, None, payoffs, mutation)), list(xs)
        )
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
