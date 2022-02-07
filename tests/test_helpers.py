import replicator
import numpy as np


def test_edges_2D():
    edges = replicator.edges_2D()

    assert len(edges) == 3
    assert isinstance(edges, np.ndarray)
    assert isinstance(edges[np.random.randint(3)], np.ndarray)


def test_edges_3D():
    edges = replicator.edges_3D()

    assert len(edges) == 4
    assert isinstance(edges, np.ndarray)
    assert isinstance(edges[np.random.randint(3)], np.ndarray)


def test_projection_2D():
    proj = replicator.projection_2D()

    assert len(proj) == 2
    assert isinstance(proj, np.ndarray)


def test_projection_3D():
    proj = replicator.projection_3D()

    assert len(proj) == 3
    assert isinstance(proj, np.ndarray)


def test_outline_2D():
    outline = replicator.outline_2D()

    assert len(outline) == 2
    assert isinstance(outline, np.ndarray)
    assert len(outline[0]) == 4


def test_outline_3D():
    outline, last_line = replicator.outline_3D()

    assert len(outline) == 3
    assert isinstance(outline, np.ndarray)

    assert len(last_line) == 3
    assert isinstance(last_line, np.ndarray)


def test_barycentric_coords_from_cartesian_2D():
    edges = replicator.edges_2D()

    expected = np.array([1, 0, 0])
    point = np.array([-1, -1 / np.sqrt(3)])
    assert (
        expected == replicator.barycentric_coords_from_cartesian(edges, point)
    ).all()

    expected = np.array([0, 1, 0])
    point = np.array([1, -1 / np.sqrt(3)])
    assert (
        expected == replicator.barycentric_coords_from_cartesian(edges, point)
    ).all()

    expected = np.array([0, 0, 1])
    point = np.array([0, 2 / np.sqrt(3)])
    assert (
        expected == replicator.barycentric_coords_from_cartesian(edges, point)
    ).all()

    expected = np.array([1 / 3, 1 / 3, 1 / 3])
    point = np.array([0, 0])
    assert (
        np.isclose(
            expected, replicator.barycentric_coords_from_cartesian(edges, point)
        )
    ).all()


def test_barycentric_coords_from_cartesian_2D():
    edges = replicator.edges_3D()

    expected = np.array([1, 0, 0, 0])
    point = np.array([-1, -1 / np.sqrt(3), -1 / np.sqrt(6)])
    assert (
        np.isclose(
            expected, replicator.barycentric_coords_from_cartesian(edges, point)
        )
    ).all()

    expected = np.array([0, 1, 0, 0])
    point = np.array([1, -1 / np.sqrt(3), -1 / np.sqrt(6)])
    assert (
        np.isclose(
            expected, replicator.barycentric_coords_from_cartesian(edges, point)
        )
    ).all()

    expected = np.array([0, 0, 1, 0])
    point = np.array([0, 2 / np.sqrt(3), -1 / np.sqrt(6)])
    assert (
        np.isclose(
            expected, replicator.barycentric_coords_from_cartesian(edges, point)
        )
    ).all()

    expected = np.array([1 / 4, 1 / 4, 1 / 4, 1 / 4])
    point = np.array([0, 0, 0])
    assert (
        np.isclose(
            expected, replicator.barycentric_coords_from_cartesian(edges, point)
        )
    ).all()
