import replicator
import numpy as np


def test_initial_conditions_edges_2D():
    num_x_points = 5

    ics = replicator.initial_conditions_edges_2D(num_x_points)

    assert ics.shape == (num_x_points * 3, 3)
    assert (np.sum(ics, axis=1) == np.ones((num_x_points * 3, 1))).all()

    # test that for each condition one of the three types is zero. By definition
    # the points on the edges mean that one type is zero.
    rows, _ = np.where(np.isclose(ics, 0))
    assert len(rows) == num_x_points * 3


def test_initial_conditions_in_simplex_2D():
    num_x_points = 5
    num_y_points = 2

    ics = replicator.initial_conditions_in_simplex_2D(
        num_x_points, num_y_points
    )

    assert ics.shape[1] == 3
    assert ics.shape[0] >= num_x_points * num_y_points
    assert (np.sum(ics, axis=1) == np.ones((len(ics), 1))).all()

    # test all initial conditions are in the simplex
    rows, _ = np.where(np.isclose(ics, 0))
    assert len(rows) == 0
