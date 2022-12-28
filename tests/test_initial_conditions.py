import replicator
import numpy as np
import Geometry3D as gm


def test_initial_conditions_edges_2D():
    num_x_points = 5

    ics = replicator.initial_conditions_edges_2D([num_x_points] * 3)

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
        [num_x_points] * 3, num_y_points
    )

    assert ics.shape[1] == 3
    assert ics.shape[0] >= num_x_points * num_y_points
    assert (np.sum(ics, axis=1) == np.ones((len(ics), 1))).all()

    # test no initial conditions are in the simplex
    rows, _ = np.where(np.isclose(ics, 0))
    assert len(rows) == 0


def test_initial_conditions_edges_3D():
    num_x_points = 5

    ics = replicator.initial_conditions_edges_3D(num_x_points)

    assert ics.shape == (num_x_points * 6, 4)
    assert (np.sum(ics, axis=1) == np.ones((num_x_points * 6, 1))).all()

    # test that for each condition two of the four types is zero.
    rows, _ = np.where(np.isclose(ics, 0))
    assert len(rows) == (num_x_points * 6) * 2


def test_initial_conditions_in_simplex_2D():
    num_x_points = 5
    num_y_points = 2
    num_z_points = 3

    ics, _ = replicator.initial_conditions_in_simplex_3D(
        num_x_points, num_y_points, num_z_points
    )

    assert ics.shape[1] == 4
    assert ics.shape[0] >= num_x_points * num_y_points * num_z_points
    assert (np.sum(ics, axis=1) == np.ones((len(ics), 1))).all()

    # test no initial conditions are in the simplex
    rows, _ = np.where(np.isclose(ics, 0, atol=10 ** -4))
    assert len(rows) == 0


def test_initial_conditions_face_A():
    tetrahedron_edges = replicator.edges_3D()
    pa = gm.Point(tetrahedron_edges[0])
    pb = gm.Point(tetrahedron_edges[1])
    pc = gm.Point(tetrahedron_edges[2])
    pd = gm.Point(tetrahedron_edges[3])

    cpg0 = gm.ConvexPolygon((pa, pb, pc))
    cpg1 = gm.ConvexPolygon((pa, pb, pd))
    cpg2 = gm.ConvexPolygon((pa, pc, pd))
    cpg3 = gm.ConvexPolygon((pb, pc, pd))

    tetrahedron = gm.ConvexPolyhedron((cpg0, cpg1, cpg2, cpg3))

    ics, _ = replicator.initial_conditions_face_A(
        tetrahedron,
        y_num=15,
    )

    assert ics.shape[1] == 4
    assert (np.sum(ics, axis=1) == np.ones((len(ics), 1))).all()

    _, cols = np.where(np.isclose(ics, 0, atol=10 ** -4))
    assert set(cols) == set([2])


def test_initial_conditions_face_B():
    tetrahedron_edges = replicator.edges_3D()
    pa = gm.Point(tetrahedron_edges[0])
    pb = gm.Point(tetrahedron_edges[1])
    pc = gm.Point(tetrahedron_edges[2])
    pd = gm.Point(tetrahedron_edges[3])

    cpg0 = gm.ConvexPolygon((pa, pb, pc))
    cpg1 = gm.ConvexPolygon((pa, pb, pd))
    cpg2 = gm.ConvexPolygon((pa, pc, pd))
    cpg3 = gm.ConvexPolygon((pb, pc, pd))

    tetrahedron = gm.ConvexPolyhedron((cpg0, cpg1, cpg2, cpg3))

    ics, _ = replicator.initial_conditions_face_B(
        tetrahedron, x_num=15, y_num=3
    )

    assert ics.shape[1] == 4
    assert (np.sum(ics, axis=1) == np.ones((len(ics), 1))).all()

    _, cols = np.where(np.isclose(ics, 0, atol=10 ** -4))
    assert set(cols) == set([0])


def test_initial_conditions_face_C():
    tetrahedron_edges = replicator.edges_3D()
    pa = gm.Point(tetrahedron_edges[0])
    pb = gm.Point(tetrahedron_edges[1])
    pc = gm.Point(tetrahedron_edges[2])
    pd = gm.Point(tetrahedron_edges[3])

    cpg0 = gm.ConvexPolygon((pa, pb, pc))
    cpg1 = gm.ConvexPolygon((pa, pb, pd))
    cpg2 = gm.ConvexPolygon((pa, pc, pd))
    cpg3 = gm.ConvexPolygon((pb, pc, pd))

    tetrahedron = gm.ConvexPolyhedron((cpg0, cpg1, cpg2, cpg3))

    ics, _ = replicator.initial_conditions_face_C(
        tetrahedron,
        z_num=5,
    )

    assert ics.shape[1] == 4
    assert (np.sum(ics, axis=1) == np.ones((len(ics), 1))).all()

    _, cols = np.where(np.isclose(ics, 0, atol=10 ** -4))
    assert set(cols) == set([1])


def test_initial_conditions_face_D():

    ics, _ = replicator.initial_conditions_face_D(x_num=5)

    assert ics.shape[1] == 4
    assert (np.sum(ics, axis=1) == np.ones((len(ics), 1))).all()

    _, cols = np.where(np.isclose(ics, 0, atol=10 ** -4))
    assert set(cols) == set([3])
