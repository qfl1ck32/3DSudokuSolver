import cv2 as cv
import numpy as np
from packages.debugging import add_circle

from packages.imaging import extract_coordinates_from_contour, order_points
from packages.imaging import show_image
from packages.logging import logger
from packages.utility import get_current_function_name

from solution.globals import hyperparameters, constants

_extract_face_if_exists_debug_index = 0


def extract_face_if_exists(face_image: np.array):
    global _extract_face_if_exists_debug_index
    _extract_face_if_exists_debug_index += 1

    debug = False
    
    if debug:
        method_name = f"{get_current_function_name()} - {_extract_face_if_exists_debug_index}"

    #

    original_image = face_image.copy()

    hsv_image = cv.cvtColor(original_image, cv.COLOR_BGR2HSV)

    black_pixels_only = ~cv.inRange(hsv_image, constants.black_low, constants.black_high)

    if debug:
        show_image(original_image, f"{method_name} Initial image")

        show_image(black_pixels_only, f"{method_name} Image after keeping black pixels")


    contours = cv.findContours(
        black_pixels_only, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]

    contour = max(contours, key=cv.contourArea)

    if debug:
        original_image_copy = original_image.copy()

        cv.drawContours(original_image_copy, [contour], -1, (0, 0, 255), 2)

        show_image(original_image_copy, f"{method_name} Contour of face")

    x, y, w, h = cv.boundingRect(contour)

    if debug:
        original_image_copy = original_image.copy()

        cv.drawContours(original_image_copy, [contour], -1, (0, 255, 0), 2)

        cv.rectangle(original_image_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)

        show_image(original_image_copy, f"{method_name} Bounding box")

    face_image = original_image[y: y + h, x: x + w]

    area = cv.contourArea(contour) / \
        (face_image.shape[0] * face_image.shape[1])

    if area < hyperparameters.face_image_extractor_cube_area_threshold:

        if debug:
            logger.info(f"{method_name} No face detected")

        return None, None

    corners = order_points(extract_coordinates_from_contour(contour), dtype=np.float32)

    if debug:
        original_image_copy = original_image.copy()

        for point in corners:
            add_circle(original_image_copy, (int(point[0]), int(point[1])), thickness=5)

        show_image(original_image_copy, f"{method_name} The corner points")

    width, height = original_image.shape[:2]

    transform_end = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    transformation_matrix = cv.getPerspectiveTransform(corners, transform_end)

    face_image = cv.warpPerspective(
        original_image, transformation_matrix, (width, height),
        borderValue=(255, 255, 255),
        borderMode=cv.BORDER_CONSTANT)

    if debug:
        show_image(face_image, f"{method_name} The final face")

     
    return cv.resize(face_image, (height, width)), corners
