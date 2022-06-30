import numpy as np
from shapely.geometry import Polygon

def get_distance(x: tuple[int, int], y: tuple[int, int]):
    return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)


def get_straight_line_equation(point_1: tuple[int, int], point_2: tuple[int, int]):
    x_point_1, y_point_1 = point_1
    x_point_2, y_point_2 = point_2

    def straight_line_equation(x: float, should_round=True):
        result = (x - x_point_1) / (x_point_2 - x_point_1) * (y_point_2 - y_point_1) + y_point_1

        return int(result) if should_round else result

    return straight_line_equation


def calculate_iou(box_1: tuple, box_2: tuple):
    poly_1 = Polygon(box_1)

    poly_2 = Polygon(box_2)

    return poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
