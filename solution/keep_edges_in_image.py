from typing import List

import numpy as np
import cv2 as cv

from packages.imaging import rotate_bound
from packages.imaging import show_image
from packages.utility import get_current_function_name

from solution.globals import hyperparameters


def keep_edges_in_image(image: np.ndarray, structure: np.ndarray, angles: List[int]):
    debug = False
    
    if debug:
        method_name = get_current_function_name()

    #

    best_answer = None
    best_angle = None
    best_struct = None

    best_percentage = 0

    for angle in angles:
        current_structure = rotate_bound(structure, angle)
        current_structure[current_structure != 255] = 0

        if debug:
            show_image(current_structure, f"{method_name} Current structure")

        closed = cv.morphologyEx(
            image, cv.MORPH_CLOSE, current_structure, iterations=1)

        thresholded = cv.threshold(
            closed, hyperparameters.keep_edges_in_image_threshold, 255, cv.THRESH_BINARY)[1]

        answer = cv.cvtColor(thresholded, cv.COLOR_GRAY2BGR)

        if debug:
            show_image(answer, f"{method_name} after close (angle: {angle})")

        content_percentage: float = np.sum(
            answer == 0) / float(answer.shape[0] * answer.shape[1])

        if content_percentage > best_percentage:
            best_percentage = content_percentage
            best_angle = angle
            best_answer = answer
            best_struct = current_structure

    return best_answer, best_angle, best_struct
