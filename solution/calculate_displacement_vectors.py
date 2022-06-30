from typing import List
from solution.globals import constants


def calculate_displacement_vectors(corners: List):
    top_left, top_right, bottom_left = corners

    horizontal_displacement = (top_left - top_right) // constants.cube_size
    vertical_displacement = (top_left - bottom_left) // constants.cube_size
   
    return horizontal_displacement, vertical_displacement
