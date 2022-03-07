from matplotlib import pyplot as plt
import numpy as np
from replicator.arrows import Annotation3D, Arrow3D
from replicator.initial_conditions import (
    initial_conditions_edges_2D,
    initial_conditions_edges_3D,
    initial_conditions_in_simplex_2D,
)
from replicator.phase_plotting import (
    plot_phase_A,
    plot_phase_B,
    plot_phase_C,
    plot_phase_D,
    plot_simplex,
)

import sympy as sym

from scipy.integrate import odeint

from .solver import fixed_points, jacobian, point_is, odes

from .helpers import (
    projection_2D,
    outline_2D,
    projection_3D,
    outline_3D,
    edges_3D,
)

import Geometry3D as gm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def _annotate3D(ax, text, xyz, *args, **kwargs):
    """Add anotation `text` to an `Axes3d` instance."""

    annotation = Annotation3D(text, xyz, *args, **kwargs)
    ax.add_artist(annotation)


def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    """Add an 3d arrow to an `Axes3D` instance."""

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


from mpl_toolkits.mplot3d.axes3d import Axes3D


setattr(Axes3D, "arrow3D", _arrow3D)

setattr(Axes3D, "annotate3D", _annotate3D)

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


def plot3D(
    payoff_mat,
    labels,
    diff_tolerance=5 * 10**-2,
    generations_forward_A=10,
    generations_backwards_A=10,
    generations_forward_B=50,
    generations_backwards_B=10,
    generations_forward_C=10,
    generations_backwards_C=10,
    generations_forward_D=10,
    generations_backwards_D=10,
    title=False
):

    assert payoff_mat.shape == (4, 4)

    fig = plt.figure(figsize=(20, 20))

    proj = projection_3D()
    edges = edges_3D()
    lines, last_line = outline_3D()
    size = payoff_mat.shape[0]

    pa = gm.Point(edges[0])
    pb = gm.Point(edges[1])
    pc = gm.Point(edges[2])
    pd = gm.Point(edges[3])

    cpg0 = gm.ConvexPolygon((pa, pb, pc))
    cpg1 = gm.ConvexPolygon((pa, pb, pd))
    cpg2 = gm.ConvexPolygon((pa, pc, pd))
    cpg3 = gm.ConvexPolygon((pb, pc, pd))

    tetrahedron = gm.ConvexPolyhedron((cpg0, cpg1, cpg2, cpg3))

    xs = np.array(sym.symbols(f"x_1:{size + 1}"))
    solutions = np.array(fixed_points(xs, payoff_mat))

    solutions = solutions[~np.any(solutions < 0, axis=1)]
    solutions = solutions[~np.any(solutions > 1, axis=1)]

    J = jacobian(xs, payoff_mat)
    point_types = [point_is(J, xs, solution) for solution in solutions]

    axes = []
    for index in [(3, 3, i) for i in range(1, 6)]:
        axes.append(fig.add_subplot(*index, projection="3d"))

    for ax in axes:
        ax.plot(lines[0], lines[1], lines[2], color="black")

        ax.plot(last_line[0], last_line[1], last_line[2], color="black")

        for label, coord, ha, va in zip(
            labels,
            [(0.0, 0.31), (0.5, 0.1), (0.85, 0.35), (0.44, 0.8)],
            ["left", "right", "right", "center"],
            ["bottom", "bottom", "bottom", "top"],
        ):
            ax.annotate(
                label,
                xy=coord,
                xycoords="axes fraction",
                ha=ha,
                va=va,
                color="black",
                weight="bold",
                fontsize=15,
            )

        for solution, point_type in zip(solutions, point_types):
            colour = colours[point_type]
            point_types.append((solution, point_type))

            ax.scatter(
                np.dot(proj, solution)[0],
                np.dot(proj, solution)[1],
                np.dot(proj, solution)[2],
                s=300,
                color="black",
                facecolor=colour,
                marker="o",
                zorder=11,
            )

    edge_ics = initial_conditions_edges_3D(4)

    time = np.linspace(0.0, 6, 100)

    for i, x in enumerate(edge_ics):

        trajectory = odeint(odes, x, time, args=(payoff_mat,))

        plot_values = np.dot(proj, trajectory.T)
        xx = plot_values[0]
        yy = plot_values[1]
        zz = plot_values[2]

        dist = np.sqrt(np.sum(np.diff(plot_values, axis=1) ** 2, axis=0))
        dist = np.cumsum(dist)

        ind = np.abs(dist - 0.05).argmin()

        for ax in axes:
            ax.arrow3D(
                xx[ind],
                yy[ind],
                zz[ind],
                xx[ind + 1] - xx[ind],
                yy[ind + 1] - yy[ind],
                zz[ind + 1] - zz[ind],
                mutation_scale=20,
                arrowstyle="-|>",
                linestyle="dashed",
                zorder=3,
                fc="black",
            )

    plot_phase_A(
        axes[0],
        tetrahedron,
        payoff_mat,
        diff_tolerance,
        generations=6,
        generations_forward=generations_forward_A,
        generations_backwards=generations_backwards_A,
    )

    plot_phase_B(
        axes[1],
        tetrahedron,
        payoff_mat,
        diff_tolerance,
        generations=6,
        generations_forward=generations_forward_B,
        generations_backwards=generations_backwards_B,
    )

    plot_phase_C(
        axes[3],
        tetrahedron,
        payoff_mat,
        diff_tolerance,
        generations=6,
        generations_forward=generations_forward_C,
        generations_backwards=generations_backwards_C,
    )

    plot_phase_D(
        axes[4],
        payoff_mat,
        diff_tolerance,
        generations=6,
        generations_forward=generations_forward_D,
        generations_backwards=generations_backwards_D,
    )

    plot_simplex(
        axes[2],
        payoff_mat,
        10**-1,
        generations=6,
        generations_forward=25,
        generations_backwards=10,
    )

    for i, idx in enumerate([[0, 1, -1], [1, 2, -1], [0, 2, -1], [0, 1, 2]]):
        if i > 1:
            i += 1
        srf = Poly3DCollection(
            [edges[idx[0]], edges[idx[1]], edges[idx[2]]],
            alpha=0.25,
            facecolor="gray",
        )

        axes[i].add_collection3d(srf)

    for ax in axes:
        ax.view_init(15, -40)

    for ax in axes:
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()

    return fig
