import math
import numpy as np
import cv2 as cv
from packages.imaging import show_image
from packages.utility import get_current_function_name

from solution.enums import SegmentType


def extract_starting_points_from_segments(image: np.ndarray, segment_type: SegmentType, segment_of_original_sizes: np.ndarray, vertical_segment_sizes: tuple = None):
    debug = False

    if debug:
        method_name = get_current_function_name()

    #

    first_path_points = []
    second_path_points = []

    gray_bitwise_not_image = cv.bitwise_not(
        cv.cvtColor(image, cv.COLOR_RGB2GRAY))

    contours, _ = cv.findContours(
        gray_bitwise_not_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if debug:
        debug_image = image.copy()

        cv.drawContours(debug_image, contours, -1, (0, 0, 255), 2)

        show_image(
            debug_image, f"{method_name} Contours, count: {len(contours)}")

    gray_bitwise_not_image = cv.cvtColor(
        gray_bitwise_not_image, cv.COLOR_GRAY2BGR)

    struct_distance = np.sqrt(
        segment_of_original_sizes.shape[0] ** 2 + segment_of_original_sizes.shape[1] ** 2)

    height, width = segment_of_original_sizes.shape[:2]

    if segment_type == SegmentType.VERTICAL:
        for contour in contours:
            top = np.array(contour[contour[:, :, 1].argmin()][0])
            bottom = np.array(contour[contour[:, :, 1].argmax()][0])

            distance = np.linalg.norm(top - bottom)

            number_of_lines = round(distance / struct_distance)

            starting_point = top

            for i in range(number_of_lines):
                dist_to_add = np.array(
                    [width, i * height])

                first_path_points.append(
                    starting_point + dist_to_add)

                second_path_points.append(
                    starting_point + dist_to_add - (width, 0))

                for point in [first_path_points[-1], second_path_points[-1]]:
                    cv.circle(i, point, 3, (0, 0, 255), 3)

    else:
        for contour in contours:
            top = np.array(contour[contour[:, :, 1].argmin()][0])
            bottom = np.array(contour[contour[:, :, 1].argmax()][0])

            distance = np.linalg.norm(top - bottom)

            number_of_lines = round(distance / struct_distance)

            prev_point = bottom if segment_type == SegmentType.SLASH else top

            sign = 1 if segment_type == SegmentType.BACKSLASH else -1

            m = np.tan(np.deg2rad(sign * 30))

            debug_image = np.array([prev_point[1] - m * prev_point[0]])

            def y(x):
                return m * x + debug_image

            for i in range(number_of_lines):
                first_path_points.append(prev_point)

                next_orientation_point = prev_point - \
                    [0, -sign * vertical_segment_sizes[1]]

                second_path_points.append(next_orientation_point)

                x = int(prev_point[0] + 0.96 * width)

                prev_point = np.array([x, int(y(x))])

    return [tuple(item) for item in first_path_points], \
        [tuple(item) for item in second_path_points]
