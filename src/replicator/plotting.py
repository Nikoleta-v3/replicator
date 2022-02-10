from matplotlib import pyplot as plt
import numpy as np
from replicator.initial_conditions import (
    initial_conditions_edges_2D,
    initial_conditions_in_simplex_2D,
)

import sympy as sym

from scipy.integrate import odeint

from .solver import fixed_points, jacobian, point_is, odes

from .helpers import projection_2D, outline_2D

colours = {"sink": "black", "source": "white", "saddle": "grey"}


def plot2D(
    payoff_mat,
    labels,
    time,
    num_of_edge_ics=5,
    num_x_points_simplex=5,
    num_y_edge_points_simplex=2,
    ax=None,
):
    assert payoff_mat.shape == (3, 3)
    if not ax:
        fig, ax = plt.subplots(figsize=(6, 6))

    size = payoff_mat.shape[0]

    xs = np.array(sym.symbols(f"x_1:{size + 1}"))
    solutions = np.array(fixed_points(xs, payoff_mat))

    solutions = solutions[~np.any(solutions < 0, axis=1)]
    solutions = solutions[~np.any(solutions > 1, axis=1)]

    proj = projection_2D()

    lines = outline_2D()

    ax.plot(
        lines[0],
        lines[1],
        clip_on=False,
        color="black",
        zorder=3,
        linewidth=0.5,
    )

    for label, coord, ha, va in zip(
        labels,
        [(0, 0), (1, 0), (0.5, 1)],
        ["right", "left", "center"],
        ["top", "top", "bottom"],
    ):
        ax.annotate(
            label,
            xy=coord,
            xycoords="axes fraction",
            ha=ha,
            va=va,
            color="black",
            weight="bold",
        )

    J = jacobian(xs, payoff_mat)
    point_types = []
    for i, solution in enumerate(solutions):
        point_type = point_is(J, xs, solution)
        colour = colours[point_type]
        point_types.append((solution, point_type))

        ax.scatter(
            np.dot(proj, solution)[0],
            np.dot(proj, solution)[1],
            s=300,
            color="black",
            facecolor=colour,
            marker="o",
            zorder=11,
        )

        if point_type == "sink":
            payoffs_of_point = np.array(
                [solution @ payoff_mat @ other for other in solutions]
            )

            if (
                len(np.where(payoffs_of_point == payoffs_of_point[i])[0]) > 0
                and len(np.where(payoffs_of_point < payoffs_of_point[i])[0]) > 0
            ):
                ax.scatter(
                    np.dot(proj, solution)[0],
                    np.dot(proj, solution)[1],
                    s=300,
                    color="black",
                    facecolor="white",
                    marker="x",
                    zorder=11,
                )

    ics_edges = initial_conditions_edges_2D(num_of_edge_ics)

    ics = initial_conditions_in_simplex_2D(
        num_x_points_simplex, num_y_edge_points_simplex
    )

    for x in np.concatenate((ics, ics_edges), axis=0):

        trajectory = odeint(odes, x, time, args=(payoff_mat,))

        plot_values = np.dot(proj, trajectory.T)
        xx = plot_values[0]
        yy = plot_values[1]

        dist = np.sqrt(np.sum(np.diff(plot_values, axis=1) ** 2, axis=0))
        dist = np.cumsum(dist)

        ind = np.abs(dist - 0.065).argmin()

        ax.plot(
            xx[: ind + 1],
            yy[: ind + 1],
            linewidth=1,
            color="black",
            zorder=3,
        )

        ax.arrow(
            xx[ind],
            yy[ind],
            xx[ind + 1] - xx[ind],
            yy[ind + 1] - yy[ind],
            shape="full",
            lw=0,
            head_width=0.03,
            edgecolor="black",
            facecolor="black",
            zorder=3,
        )

    ax.set_xticks([], minor=False)
    ax.set_yticks([], minor=False)
    ax.set_aspect(1)

    ax.axis("off")

    information = [f"There are a total of {len(point_types)} fixed points."]
    for point, type_ in point_types:
        information.append(f"{point} is a {type_} point.")

    return ax, information
