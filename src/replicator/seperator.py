import replicator
import numpy as np
import itertools
from scipy.integrate import odeint


def grib2D(pointA, pointB, n=10):

    proj = replicator.projection_2D()
    coordA = np.dot(proj, pointA.T)
    coordB = np.dot(proj, pointB.T)

    y = max(coordA[1], coordB[1])
    x = min(coordA[0], coordB[0])
    coordC = (x, y)

    xs = np.linspace(coordC[0], max(coordA[0], coordB[0]), n)
    ys = np.linspace(coordC[1], min(coordA[1], coordB[1]), n)

    return xs, ys


def get_directions(pointA, pointB, matrix, n=10, sforward=20):

    edges = replicator.edges_2D()
    directions_dict = {
        tuple(comb): i
        for i, comb in enumerate(list(itertools.product([0, 1], repeat=3)))
    }
    directions = np.zeros((n, n))
    xs, ys = grib2D(pointA, pointB, n)

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            time = np.linspace(0.0, sforward, 100)
            start = replicator.barycentric_coords_from_cartesian(edges, (x, y))
            trajectory = odeint(replicator.odes, start, time, args=(matrix,))
            directions[i, j] = directions_dict[
                tuple(
                    [
                        1 if c > 0 else 0
                        for c in (trajectory[0] - trajectory[-1])
                    ]
                )
            ]
    return directions, xs, ys


def estimate_line(directions, xs, ys, valA, valB):

    Is, Js = np.where(directions == valA)
    n = directions.shape[0] - 1
    rhs = [(i, j) for i, j in zip(Is, Js) if directions[i, min(n, j + 1)] == valB]
    toplot = [((xs[i] + xs[i]) / 2, (ys[j] + ys[min(n, j + 1)]) / 2) for i, j in rhs]

    X, Y = zip(*toplot)
    a, b = np.polyfit(X, Y, 1)

    return a * np.array(X) + b, X, Y


def grib3D(pointA, pointB, pointC, pointD, n=10):
    zs = np.linspace(pointB[2], pointA[2], n)
    xs = np.linspace(pointB[0], pointC[0], n + 2)
    ys = np.linspace(pointC[1], pointD[1], n + 2)
    return xs[1:-1], ys[1:-1], zs


def get_coordinates_of_3D_point(point):
    proj = replicator.projection_3D()
    return np.dot(proj, point.T)


def get_directions3D(pointA, pointB, pointC, pointD, matrix, n=10, sforward=20):

    edges = replicator.edges_3D()
    directions_dict = {
        tuple(comb): i
        for i, comb in enumerate(list(itertools.product([0, 1], repeat=4)))
    }
    directions = np.zeros((n, n, n))
    xs, ys, zs = grib3D(pointA, pointB, pointC, pointD, n)

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            for l, z in enumerate(zs):
                time = np.linspace(0.0, sforward, 100)
                start = replicator.barycentric_coords_from_cartesian(
                    edges, (x, y, z)
                )
                if (start < 0).any():
                    directions[i, j, l] = -1
                else:
                    trajectory = odeint(
                        replicator.odes, start, time, args=(matrix,)
                    )
                    directions[i, j, l] = directions_dict[
                        tuple(
                            [
                                1 if c > 0 else 0
                                for c in (trajectory[0] - trajectory[-1])
                            ]
                        )
                    ]
    return directions, xs, ys, zs


def fit_Plane_with_LTSQ(XYZ):
    (rows, cols) = XYZ.shape
    G = np.ones((rows, 3))
    G[:, 0] = XYZ[:, 0]
    G[:, 1] = XYZ[:, 1]
    Z = XYZ[:, 2]
    (a, b, c), _, _, _ = np.linalg.lstsq(G, Z)
    normal = (a, b, -1)
    nn = np.linalg.norm(normal)
    normal = normal / nn
    return (c, normal)


def estimate_plane(directions, xs, ys, zs, valA, valB, maxx=None):

    Is, Js, Ls = np.where(directions == valA)
    n = directions.shape[0] - 1
    rh = [
        (i, j, l)
        for i, j, l in zip(Is, Js, Ls)
        if (directions[i, min([n, j + 1]), l] == valB)
    ]

    toplot = [
        (xs[i], (ys[j] + ys[min([n, j + 1])]) / 2, zs[l]) for i, j, l in rh
    ]

    ow = [
        (i, j, l)
        for i, j, l in zip(Is, Js, Ls)
        if (directions[i, j, min([n, l + 1])] == valB)
    ]

    toplot += [
        (xs[i], ys[j], (zs[l] + zs[min([n, l + 1])]) / 2) for i, j, l in ow
    ]

    data = np.array(toplot)
    c, normal = fit_Plane_with_LTSQ(data)

    if maxx == None:
        maxx = np.max(data[:, 0])
    else:
        maxx =maxx
    maxy = np.max(data[:, 1])
    minx = np.min(data[:, 0])
    miny = np.min(data[:, 1])

    point = np.array([0.0, 0.0, c])
    d = -point.dot(normal)

    XX, YY = np.meshgrid([minx, maxx], [miny, maxy])
    Z = (-normal[0] * XX - normal[1] * YY - d) * 1.0 / normal[2]

    return XX, YY, Z, toplot
