from .solver import odes, fixed_points, jacobian, point_is
from .helpers import (
    edges_2D,
    edges_3D,
    projection_2D,
    projection_3D,
    outline_2D,
    outline_3D,
    barycentric_coords_from_cartesian,
)

from .initial_conditions import (
    initial_conditions_edges_2D,
    initial_conditions_in_simplex_2D,
    initial_conditions_edges_3D,
    initial_conditions_in_simplex_3D,
    initial_conditions_face_A,
    initial_conditions_face_B,
    initial_conditions_face_C,
    initial_conditions_face_D,
)

from .plotting import plot2D
