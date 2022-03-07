from .helpers import edges_3D, barycentric_coords_from_cartesian, projection_3D
from .initial_conditions import (
    initial_conditions_in_simplex_3D,
    initial_conditions_face_A,
    initial_conditions_face_B,
    initial_conditions_face_C,
    initial_conditions_face_D,
)
from .solver import odes

from .grouping import (
    sinks_of_initial_conditions,
    groups_ics_based_on_sinks,
    min_distance_point,
)

from scipy.integrate import odeint


import numpy as np


def plot_simplex(
    ax,
    payoff_mat,
    diff_tolerance,
    generations=6,
    generations_forward=10,
    generations_backwards=10,
    steps=100,
):

    proj = projection_3D()
    tetrahedron_edges = edges_3D()
    initial_conditions, points = initial_conditions_in_simplex_3D(3, 3, 3)

    directions = sinks_of_initial_conditions(
        initial_conditions, payoff_mat, diff_tolerance, generations
    )

    groups = groups_ics_based_on_sinks(directions)

    to_plot = []
    for group in groups:
        to_plot.append(
            min_distance_point(
                points[list(group), :],
            )
        )

    for matrix, gen in zip(
        [payoff_mat, -payoff_mat], [generations_forward, generations_backwards]
    ):
        for point in to_plot:
            new_gen = 0
            time = np.linspace(0.0, gen, steps)
            start = barycentric_coords_from_cartesian(tetrahedron_edges, point)
            trajectory = odeint(odes, start, time, args=(matrix,))

            # while (
            #     np.isclose(trajectory[-1], trajectory[-2], 10**-5).all()
            #     == False
            # ) and (np.isclose(sum(trajectory[-1]), 1)):
            #     new_gen += 1
            #     new_trajectory = odeint(
            #         odes, trajectory[-1], time, args=(matrix,)
            #     )

            # trajectory = np.concatenate((trajectory, new_trajectory))

            plot_values = np.dot(proj, trajectory.T)
            xx = plot_values[0]
            yy = plot_values[1]
            zz = plot_values[2]

            ax.plot3D(xx, yy, zz, color="gray", linewidth=3)
    return ax


def plot_phase_A(
    ax,
    tetrahedron,
    payoff_mat,
    diff_tolerance,
    steps=100,
    generations=6,
    generations_forward=10,
    generations_backwards=10,
):

    proj = projection_3D()
    tetrahedron_edges = edges_3D()
    initial_conditions_A, points = initial_conditions_face_A(
        tetrahedron, y_num=7
    )

    directions = sinks_of_initial_conditions(
        initial_conditions_A, payoff_mat, diff_tolerance, generations
    )

    groups = groups_ics_based_on_sinks(directions)

    to_plot = []
    for group in groups:
        to_plot.append(
            min_distance_point(
                points[list(group), :],
            )
        )

    for matrix, gen in zip(
        [payoff_mat, -payoff_mat], [generations_forward, generations_backwards]
    ):
        for point in to_plot:
            new_gen = 0
            time = np.linspace(0.0, gen, steps)
            start = barycentric_coords_from_cartesian(tetrahedron_edges, point)
            trajectory = odeint(odes, start, time, args=(matrix,))

            # while (
            #     np.isclose(trajectory[-1], trajectory[-2], 10**-5).all()
            #     == False
            # ) and (np.isclose(sum(trajectory[-1]), 1)):
            #     new_gen += 1
            #     new_trajectory = odeint(
            #         odes, trajectory[-1], time, args=(matrix,)
            #     )

            #     trajectory = np.concatenate((trajectory, new_trajectory))

            plot_values = np.dot(proj, trajectory.T)
            xx = plot_values[0]
            yy = plot_values[1]
            zz = plot_values[2]

            ax.plot3D(xx, yy, zz, color="gray", linewidth=3)
    return ax


def plot_phase_B(
    ax,
    tetrahedron,
    payoff_mat,
    diff_tolerance,
    steps=100,
    generations=6,
    generations_forward=10,
    generations_backwards=10,
):

    proj = projection_3D()
    tetrahedron_edges = edges_3D()
    initial_conditions_B, points = initial_conditions_face_B(
        tetrahedron, x_num=5
    )

    directions = sinks_of_initial_conditions(
        initial_conditions_B, payoff_mat, diff_tolerance, generations
    )

    groups = groups_ics_based_on_sinks(directions)

    to_plot = []
    for group in groups:
        to_plot.append(
            min_distance_point(
                points[list(group), :],
            )
        )

    for matrix, gen in zip(
        [payoff_mat, -payoff_mat], [generations_forward, generations_backwards]
    ):
        for point in to_plot:
            new_gen = 0
            time = np.linspace(0.0, gen, steps)
            start = barycentric_coords_from_cartesian(tetrahedron_edges, point)
            trajectory = odeint(odes, start, time, args=(matrix,))

            # while (
            #     np.isclose(trajectory[-1], trajectory[-2], 10**-5).all()
            #     == False
            # ) and (np.isclose(sum(trajectory[-1]), 1)):
            #     new_gen += 1
            #     new_trajectory = odeint(
            #         odes, trajectory[-1], time, args=(matrix,)
            #     )

            #     trajectory = np.concatenate((trajectory, new_trajectory))

            plot_values = np.dot(proj, trajectory.T)
            xx = plot_values[0]
            yy = plot_values[1]
            zz = plot_values[2]

            ax.plot3D(xx, yy, zz, color="gray", linewidth=3)
    return ax


def plot_phase_C(
    ax,
    tetrahedron,
    payoff_mat,
    diff_tolerance,
    steps=100,
    generations=6,
    generations_forward=10,
    generations_backwards=10,
):

    proj = projection_3D()
    tetrahedron_edges = edges_3D()
    initial_conditions_C, points = initial_conditions_face_C(
        tetrahedron, z_num=5
    )

    directions = sinks_of_initial_conditions(
        initial_conditions_C, payoff_mat, diff_tolerance, generations
    )

    groups = groups_ics_based_on_sinks(directions)

    to_plot = []
    for group in groups:
        to_plot.append(
            min_distance_point(
                points[list(group), :],
            )
        )

    for matrix, gen in zip(
        [payoff_mat, -payoff_mat], [generations_forward, generations_backwards]
    ):
        for point in to_plot:
            new_gen = 0
            time = np.linspace(0.0, gen, steps)
            start = barycentric_coords_from_cartesian(tetrahedron_edges, point)
            trajectory = odeint(odes, start, time, args=(matrix,))

            # while (
            #     np.isclose(trajectory[-1], trajectory[-2], 10**-5).all()
            #     == False
            # ) and (np.isclose(sum(trajectory[-1]), 1)):
            #     new_gen += 1
            #     new_trajectory = odeint(
            #         odes, trajectory[-1], time, args=(matrix,)
            #     )

            #     trajectory = np.concatenate((trajectory, new_trajectory))

            plot_values = np.dot(proj, trajectory.T)
            xx = plot_values[0]
            yy = plot_values[1]
            zz = plot_values[2]

            ax.plot3D(xx, yy, zz, color="gray", linewidth=3)
    return ax


def plot_phase_D(
    ax,
    payoff_mat,
    diff_tolerance,
    steps=100,
    generations=6,
    generations_forward=10,
    generations_backwards=10,
):

    proj = projection_3D()
    tetrahedron_edges = edges_3D()
    initial_conditions_D, points = initial_conditions_face_D(x_num=5)

    directions = sinks_of_initial_conditions(
        initial_conditions_D, payoff_mat, diff_tolerance, generations
    )

    groups = groups_ics_based_on_sinks(directions)

    to_plot = []
    for group in groups:
        to_plot.append(
            min_distance_point(
                points[list(group), :],
            )
        )

    for matrix, gen in zip(
        [payoff_mat, -payoff_mat], [generations_forward, generations_backwards]
    ):
        for point in to_plot:
            new_gen = 0
            time = np.linspace(0.0, gen, steps)
            start = barycentric_coords_from_cartesian(tetrahedron_edges, point)
            trajectory = odeint(odes, start, time, args=(matrix,))

            # while (
            #     np.isclose(trajectory[-1], trajectory[-2], 10**-5).all()
            #     == False
            # ) and (np.isclose(sum(trajectory[-1]), 1)):
            #     new_gen += 1
            #     new_trajectory = odeint(
            #         odes, trajectory[-1], time, args=(matrix,)
            #     )

            #     trajectory = np.concatenate((trajectory, new_trajectory))

            plot_values = np.dot(proj, trajectory.T)
            xx = plot_values[0]
            yy = plot_values[1]
            zz = plot_values[2]

            ax.plot3D(xx, yy, zz, color="gray", linewidth=3)
    return ax
