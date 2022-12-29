from .solver import (
    odes,
    fixed_points,
    jacobian,
    point_is,
    odes_for_numerical_solver,
)
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
    face_function,
)

from .plotting import (
    plot2D,
    plot3D,
    plot3D_exterior,
    plot2D_exterior,
    initial_conditions_edges_3D_tweaked,
    arrow_coordinates_in_3D,
)


from .arrows import Annotation3D, Arrow3D

from .grouping import (
    min_distance_point,
    sinks_of_initial_conditions,
    groups_ics_based_on_sinks,
)


from .phase_plotting import (
    plot_phase_A,
    plot_phase_B,
    plot_phase_C,
    plot_phase_D,
    plot_simplex,
)

from .seperator import *