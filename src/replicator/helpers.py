"""Files contains functions that help with coordinates."""

import numpy as np


def edges_2D():
    return np.array(
        [
            np.array([-1, -1 / np.sqrt(3)]),
            np.array([1, -1 / np.sqrt(3)]),
            np.array([0, 2 / np.sqrt(3)]),
        ]
    )


def projection_2D():
    return np.array(
        [[-1, 1, 0], [-1 / np.sqrt(3), -1 / np.sqrt(3), 2 / np.sqrt(3)]]
    )


def outline_2D():
    return np.array(
        [
            [-1, 1, 0, -1],
            [-1 / np.sqrt(3), -1 / np.sqrt(3), 2 / np.sqrt(3), -1 / np.sqrt(3)],
        ]
    )


def edges_3D():
    return np.array(
        [
            [-1, -1 / np.sqrt(3), -1 / np.sqrt(6)],
            [1, -1 / np.sqrt(3), -1 / np.sqrt(6)],
            [0, 2 / np.sqrt(3), -1 / np.sqrt(6)],
            [0, 0, 3 / np.sqrt(6)],
        ]
    )


def projection_3D():
    return np.array(
        [
            [-1, 1, 0, 0],
            [-1 / np.sqrt(3), -1 / np.sqrt(3), 2 / np.sqrt(3), 0],
            [-1 / np.sqrt(6), -1 / np.sqrt(6), -1 / np.sqrt(6), 3 / np.sqrt(6)],
        ]
    )


def outline_3D():
    lines = np.array(
        [
            [1, 0, -1, 1, 0, -1],
            [
                -1 / np.sqrt(3),
                2 / np.sqrt(3),
                -1 / np.sqrt(3),
                -1 / np.sqrt(3),
                0,
                -1 / np.sqrt(3),
            ],
            [
                -1 / np.sqrt(6),
                -1 / np.sqrt(6),
                -1 / np.sqrt(6),
                -1 / np.sqrt(6),
                3 / np.sqrt(6),
                -1 / np.sqrt(6),
            ],
        ]
    )

    last_line = np.array(
        [
            [0, 0],
            [0, 2 / np.sqrt(3)],
            [3 / np.sqrt(6), -1 / np.sqrt(6)],
        ]
    )
    return lines, last_line


def barycentric_coords_from_cartesian(edges, point):

    T = (np.array(edges[:-1]) - edges[-1]).T
    v = np.dot(np.linalg.inv(T), np.array(point) - edges[-1])
    v.resize(len(edges))
    v[-1] = 1 - v.sum()
    v = (v.T / np.sum(v)).T
    return v
