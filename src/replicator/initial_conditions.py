import numpy as np
from .helpers import edges_2D, barycentric_coords_from_cartesian
import itertools
import shapely.geometry as shp


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

    # use this difference from various x points
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


# # def initial_conditions_edges_3D(num_points):
# #     ics = np.zeros([num_points * 6, 4])
# #     tetrahedron_edges = edges_3D()

# #     for i, point in enumerate(itertools.combinations(tetrahedron_edges, r=2)):
# #         mx = point[1][0] - point[0][0]
# #         my = point[1][1] - point[0][1]
# #         mz = point[1][2] - point[0][2]

# #         for k, j in enumerate(np.linspace(0, 1, num_points)):
# #             x = point[0][0] + mx * j
# #             y = point[0][1] + my * j
# #             z = point[0][2] + mz * j

# #             ics[i * num_points + k, :] = barycentric_coords_from_cartesian(
# #                 tetrahedron_edges, [x, y, z]
# #             )

# #     return ics
