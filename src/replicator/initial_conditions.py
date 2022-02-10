import numpy as np
from .helpers import (
    edges_2D,
    barycentric_coords_from_cartesian,
    edges_3D,
    face_function,
)
import itertools
import shapely.geometry as shp

import Geometry3D as gm


def initial_conditions_edges_2D(num_x_points):
    """This function calculates the linear lines between each pair of edges
    and then draws equally spaced points on these lines.

    Parameters
    ----------
    num_x_points : int
        Number of points to draw on each line of the triangle.

    Returns
    -------
    np.array
        The barycentric coordinates of the initial conditions on the line of the triangle.
    """
    ics = np.zeros([num_x_points * 3, 3])
    edges = edges_2D()

    for i, point in enumerate(itertools.combinations(edges, r=2)):
        x = np.array([point[0][0], point[1][0]])
        y = np.array([point[0][1], point[1][1]])
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]

        xs = np.linspace(point[0][0], point[1][0], num_x_points + 2)[1:-1]
        for j, x in enumerate(xs):
            point = (x, x * m + c)
            ics[i * num_x_points + j, :] = barycentric_coords_from_cartesian(
                edges, point
            )

    return ics


def initial_conditions_in_simplex_2D(num_x_points, num_y_edge_points):
    """Returns initial conditions in the 2D simplex. Note that edges are not
    included."""
    edges = edges_2D()
    poly = shp.Polygon(edges)

    min_x, min_y, max_x, max_y = poly.bounds
    xs = np.linspace(int(min_x), int(max_x), num_x_points + 2)[1:-1]

    # estimate diff between y-points
    x = xs[0]
    x_line = shp.LineString([(x, min_y), (x, max_y)])
    x_line_intercept_min, x_line_intercept_max = (
        x_line.intersection(poly).xy[1].tolist()
    )
    yy = np.linspace(
        x_line_intercept_min, x_line_intercept_max, num_y_edge_points + 2
    )

    points = [[x, y] for y in yy[1:-1]]
    difference_in_y = yy[1] - yy[0]

    # use this difference for various x points
    for x in xs[1:]:
        x_line = shp.LineString([(x, min_y), (x, max_y)])
        x_line_intercept_min, x_line_intercept_max = (
            x_line.intersection(poly).xy[1].tolist()
        )
        yy = np.arange(
            x_line_intercept_min, x_line_intercept_max, difference_in_y
        )

        for y in yy[1:-1]:
            points.append([x, y])

    ics = np.zeros([len(points), 3])
    for i, point in enumerate(points):
        ics[i, :] = barycentric_coords_from_cartesian(edges, point)

    return ics


def initial_conditions_edges_3D(num_points):
    num_points = num_points + 2
    ics = np.zeros([num_points * 6, 4])
    tetrahedron_edges = edges_3D()

    for i, point in enumerate(itertools.combinations(tetrahedron_edges, r=2)):
        mx = point[1][0] - point[0][0]
        my = point[1][1] - point[0][1]
        mz = point[1][2] - point[0][2]

        for k, j in enumerate(np.linspace(0, 1, num_points)):
            x = point[0][0] + mx * j
            y = point[0][1] + my * j
            z = point[0][2] + mz * j

            ics[i * num_points + k, :] = barycentric_coords_from_cartesian(
                tetrahedron_edges, [x, y, z]
            )

    for check in [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]:
        to_del = (np.isclose(ics, check)).all(axis=1).nonzero()
        ics = np.delete(ics, to_del, axis=0)

    return ics


def initial_conditions_in_simplex_3D(
    num_x_points, num_y_edge_points, num_z_edge_points
):
    """Returns initial conditions in the 2D simplex. Note that edges are not
    included."""
    edges = edges_2D()
    tetrahedron_edges = edges_3D()
    pa = gm.Point(tetrahedron_edges[0])
    pb = gm.Point(tetrahedron_edges[1])
    pc = gm.Point(tetrahedron_edges[2])
    pd = gm.Point(tetrahedron_edges[3])

    cpg0 = gm.ConvexPolygon((pa, pb, pc))
    cpg1 = gm.ConvexPolygon((pa, pb, pd))
    cpg2 = gm.ConvexPolygon((pa, pc, pd))
    cpg3 = gm.ConvexPolygon((pb, pc, pd))

    tetrahedron = gm.ConvexPolyhedron((cpg0, cpg1, cpg2, cpg3))
    poly = shp.Polygon(edges)

    min_x, min_y, max_x, max_y = poly.bounds
    xs = np.linspace(int(min_x), int(max_x), num_x_points + 2)[1:-1]

    # estimate diff between y-points
    x = xs[0]
    x_line = shp.LineString([(x, min_y), (x, max_y)])
    x_line_intercept_min, x_line_intercept_max = (
        x_line.intersection(poly).xy[1].tolist()
    )
    yy = np.linspace(
        x_line_intercept_min, x_line_intercept_max, num_y_edge_points + 2
    )

    points = [[x, y, (-1 / np.sqrt(6))] for y in yy[1:-1]]
    difference_in_y = yy[1] - yy[0]

    # use this difference for various x points
    for x in xs[1:]:
        x_line = shp.LineString([(x, min_y), (x, max_y)])
        x_line_intercept_min, x_line_intercept_max = (
            x_line.intersection(poly).xy[1].tolist()
        )
        yy = np.arange(
            x_line_intercept_min, x_line_intercept_max, difference_in_y
        )

        for y in yy[1:-1]:
            points.append([x, y, (-1 / np.sqrt(6))])

    tetra_points = []

    # estimate diff between z-points
    x, y, z = points[0]
    x_line = gm.Segment(gm.Point(x, y, z), gm.Point(x, y, (3 / np.sqrt(6))))
    intersection = gm.intersection(tetrahedron, x_line)
    z_line_intercept_min, z_line_intercept_max = sorted(
        [intersection[0][-1], intersection[1][-1]]
    )
    zz = np.linspace(
        z_line_intercept_min, z_line_intercept_max, num_z_edge_points + 2
    )
    for z in zz[1:-1]:
        tetra_points.append([x, y, z])
    diff = zz[1] - zz[0]

    for x, y, _ in points[1:]:

        x_line = gm.Segment(
            gm.Point(x, y, (-1 / np.sqrt(6))), gm.Point(x, y, 3 / np.sqrt(6))
        )

        intersection = gm.intersection(tetrahedron, x_line)

        if isinstance(list(intersection)[0], float):
            tetra_points.append(list(intersection))

        else:

            z_line_intercept_min, z_line_intercept_max = sorted(
                [intersection[0][-1], intersection[1][-1]]
            )

            zz = np.arange(z_line_intercept_min, z_line_intercept_max, diff)

            for z in zz[1:-1]:
                tetra_points.append([x, y, z])

    ics = np.zeros([len(tetra_points), 4])
    for i, point in enumerate(tetra_points):
        ics[i, :] = barycentric_coords_from_cartesian(tetrahedron_edges, point)

    return ics, tetra_points


def initial_conditions_face_A(tetrahedron, y_num=15, grid_size=0.13):

    tetra_edges = edges_3D()

    a, b, c, d = face_function(tetra_edges[0], tetra_edges[1], tetra_edges[-1])

    pts = np.zeros((y_num - 2, 3))

    xs = 0
    ys = np.linspace(0, min(tetra_edges[:, 1]), y_num)
    for i, y in enumerate(ys[1:-1]):
        z = (a * xs + b * y - d) / -c
        pts[i, 0] = xs
        pts[i, 1] = y
        pts[i, 2] = z

    points = []
    for i, pt in enumerate(pts):
        x_line = gm.Segment(
            gm.Point(-1, pt[1], pt[2]), gm.Point(1, pt[1], pt[2])
        )
        intersection = gm.intersection(tetrahedron, x_line)
        x_line_intercept_min, x_line_intercept_max = sorted(
            [intersection[0][0], intersection[1][0]]
        )
        n_sample = int(
            np.ceil(
                np.abs(x_line_intercept_max - x_line_intercept_min) / grid_size
            )
        )

        xx = np.linspace(x_line_intercept_min, x_line_intercept_max, n_sample)
        for x in xx[1:-1]:
            points.append((x, pt[1], pt[2]))

        ics = np.zeros([len(points), 4])
        for i, point in enumerate(points):
            ics[i, :] = barycentric_coords_from_cartesian(tetra_edges, point)

    return ics, points


def initial_conditions_face_B(tetrahedron, x_num=15, grid_size=0.13):

    tetra_edges = edges_3D()

    a, b, c, d = face_function(tetra_edges[1], tetra_edges[2], tetra_edges[-1])

    pts = np.zeros((x_num - 2, 3))

    ys = 0.1
    xs = np.linspace(0, 0.65, x_num)
    for i, x in enumerate(xs[1:-1]):

        z = (a * x + b * ys - d) / -c

        pts[i, 0] = x
        pts[i, 1] = ys
        pts[i, 2] = z

    points = []
    for i, pt in enumerate(pts):
        x_line = gm.Segment(
            gm.Point(
                (c * pt[2] + b * min(tetra_edges[:, 1]) - d) / -a,
                min(tetra_edges[:, 1]),
                pt[2],
            ),
            gm.Point(
                (c * pt[2] + b * max(tetra_edges[:, 1]) - d) / -a,
                max(tetra_edges[:, 1]),
                pt[2],
            ),
        )

        intersection = gm.intersection(tetrahedron, x_line)

        y_line_intercept_min, y_line_intercept_max = sorted(
            [intersection[0][1], intersection[1][1]]
        )

        n_sample = int(
            np.ceil(
                np.abs(y_line_intercept_max - y_line_intercept_min) / grid_size
            )
        )

        yy = np.linspace(y_line_intercept_min, y_line_intercept_max, n_sample)

        for y in yy[1:-1]:
            points.append(((c * pt[2] + b * y - d) / -a, y, pt[2]))

        ics = np.zeros([len(points), 4])
        for i, point in enumerate(points):
            ics[i, :] = barycentric_coords_from_cartesian(tetra_edges, point)

    return ics, points


def initial_conditions_face_C(tetrahedron, x_num=15, grid_size=0.13):

    tetra_edges = edges_3D()

    a, b, c, d = face_function(tetra_edges[0], tetra_edges[2], tetra_edges[-1])

    pts = np.zeros((x_num - 2, 3))

    ys = 0.1
    xs = np.linspace(-0.65, 0, x_num)
    for i, x in enumerate(xs[1:-1]):

        z = (a * x + b * ys - d) / -c
        pts[i, 0] = x
        pts[i, 1] = ys
        pts[i, 2] = z

    points = []
    for i, pt in enumerate(pts):
        x_line = gm.Segment(
            gm.Point(
                (c * pt[2] + b * min(tetra_edges[:, 1]) - d) / -a,
                min(tetra_edges[:, 1]),
                pt[2],
            ),
            gm.Point(
                (c * pt[2] + b * max(tetra_edges[:, 1]) - d) / -a,
                max(tetra_edges[:, 1]),
                pt[2],
            ),
        )

        intersection = gm.intersection(tetrahedron, x_line)

        y_line_intercept_min, y_line_intercept_max = sorted(
            [intersection[0][1], intersection[1][1]]
        )

        n_sample = int(
            np.ceil(
                np.abs(y_line_intercept_max - y_line_intercept_min) / grid_size
            )
        )

        yy = np.linspace(y_line_intercept_min, y_line_intercept_max, n_sample)

        for y in yy[1:-1]:
            points.append(((c * pt[2] + b * y - d) / -a, y, pt[2]))

    ics = np.zeros([len(points), 4])
    for i, point in enumerate(points):
        ics[i, :] = barycentric_coords_from_cartesian(tetra_edges, point)

    return ics, points


def initial_conditions_face_D(grid_size=0.13):

    edges = edges_2D()

    tetra_edges = edges_3D()

    z = tetra_edges[0][-1]

    poly = shp.Polygon(edges)

    min_x, min_y, max_x, max_y = poly.bounds

    n = int(np.ceil(np.abs(max_x - min_x) / grid_size))
    xs = np.linspace(int(min_x), int(max_x), n)[1:-1]

    points = []
    for x in xs[1:-1]:

        x_line = shp.LineString([(x, min_y), (x, max_y)])
        x_line_intercept_min, x_line_intercept_max = (
            x_line.intersection(poly).xy[1].tolist()
        )

        n_sample = int(
            np.ceil(
                np.abs(x_line_intercept_max - x_line_intercept_min) / grid_size
            )
        )
        yy = np.linspace(x_line_intercept_min, x_line_intercept_max, n_sample)

        for y in yy[1:-1]:
            points.append([x, y, z])

    ics = np.zeros([len(points), 4])
    for i, point in enumerate(points):
        ics[i, :] = barycentric_coords_from_cartesian(tetra_edges, point)

    return ics, points
