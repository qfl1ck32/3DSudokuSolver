import numpy as np

from packages.geometry import get_distance, get_straight_line_equation

from solution.enums import Axis

from solution.globals import hyperparameters

def add_padding_to_coordinates(coordinates: np.array, axis: Axis):
    padding_amount = hyperparameters.padding_amount[axis]

    a, b, c, d = coordinates

    if axis == Axis.Z:
        ac = get_distance(a, c)

        bd = get_distance(b, d)

        rep = ac / bd

        to_add = int(padding_amount / rep)

        a[0] -= padding_amount
        c[0] += padding_amount

        b[1] -= to_add
        d[1] += to_add

    else:
        ac_y_eq = get_straight_line_equation(a, c)
        bd_y_eq = get_straight_line_equation(b, d)

        a[0] -= padding_amount
        a[1] = ac_y_eq(a[0])

        c[0] += padding_amount
        c[1] = ac_y_eq(c[0])

        b[0] += padding_amount
        b[1] = bd_y_eq(b[0])

        d[0] -= padding_amount
        d[1] = bd_y_eq(d[0])

    return np.array([a, b, c, d], dtype=np.float32)
