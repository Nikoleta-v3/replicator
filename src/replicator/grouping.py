import numpy as np

from .solver import odes
import copy
from scipy.integrate import odeint


def sinks_of_initial_conditions(ics, payoff_mat, diff_tolerance, generations=6):
    steps = 100
    size = len(ics)
    directions = np.zeros((size, size))

    Is, Js = np.triu_indices(n=size)
    for i, j in zip(Is, Js):
        if i == j:
            directions[i, j] = 1
        else:
            run_generations = generations
            time = np.linspace(0.0, generations, steps)
            sinks = []
            for start in [ics[i], ics[j]]:

                trajectory = odeint(odes, start, time, args=(payoff_mat,))

                while (
                    np.isclose(
                        trajectory[-1], trajectory[-2], atol=10**-3
                    ).all()
                    == False
                ):
                    run_generations += generations - 4
                    trajectory = odeint(
                        odes, trajectory[-1], time, args=(payoff_mat,)
                    )

                sinks.append(trajectory[-1])

            if np.isclose(*sinks, atol=diff_tolerance).all() == True:
                directions[i, j] = 1
    return directions


def groups_ics_based_on_sinks(directions):

    plotting_groups = []
    groups = [np.where(row == 1)[0] for row in directions]
    copy_group = copy.deepcopy(groups)

    while len(copy_group) > 0:
        group = copy_group[0]
        to_exclude = [0]
        for i, other_group in enumerate(copy_group[1:]):
            if set(group) & set(other_group):
                group = set(group) | set(other_group)

                to_exclude.append(i + 1)
        plotting_groups.append(group)
        copy_group = [
            group for i, group in enumerate(copy_group) if i not in to_exclude
        ]

    return plotting_groups


def min_distance_point(points):

    size = len(points)
    distances = np.zeros((size, size))
    Is, Js = np.triu_indices(n=size)
    for i, j in zip(Is, Js):
        squared_dist = np.sum(
            (np.array(points[i]) - np.array(points[j])) ** 2, axis=0
        )
        distances[i, j] = np.sqrt(squared_dist)

    distances = distances + distances.T - np.diag(distances.diagonal())

    return points[np.argmin(np.sum(distances, axis=0))]
