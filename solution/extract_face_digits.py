import cv2 as cv
import numpy as np

from packages.imaging import rotate_bound, show_image
from packages.logging import logger
from packages.utility import get_current_function_name

from solution.globals import constants
from solution.enums import Axis

from packages.imaging import sliding_window

from solution.types import SudokuFace

_extract_face_digits_debug_index = 0

def extract_face_digits(face: SudokuFace):
    global _extract_face_digits_debug_index
    _extract_face_digits_debug_index += 1

    debug = False

    if debug:
        method_name = f"{get_current_function_name()} - {_extract_face_digits_debug_index}"
    #

    result = []

    if debug:
        show_image(face.image, f"{method_name} Image before processing")

    image = cv.cvtColor(face.image, cv.COLOR_BGR2GRAY)

    image = cv.threshold(image, 160, 255, cv.THRESH_BINARY_INV)[1]

    if debug:
        show_image(image, f"{method_name} Image after gray & threshold")


    edges_vertical = cv.morphologyEx(image, cv.MORPH_OPEN, constants.extract_face_digits_element_vertical)
    edges_horizontal = cv.morphologyEx(
        image, cv.MORPH_OPEN, constants.extract_face_digits_element_horizontal)

    edges = edges_vertical | edges_horizontal

    if debug:
        show_image(edges, f"{method_name} Edges")

    image = image & ~edges

    if debug:
        show_image(image, f"{method_name} Image after removing lines")
        

    image = cv.morphologyEx(image, cv.MORPH_OPEN, constants.extract_face_digits_element_noise)

    if debug:
        show_image(image, f"{method_name} Image after removing noise")

    image = cv.resize(image, (image.shape[1], image.shape[1]))

    face_shape = image.shape

    cube_length = int(face_shape[0] / 3)

    if debug:
        show_image(image, f"{method_name} Image after processing")
    
    for x, y, cell in sliding_window(image, cube_length, (cube_length, cube_length)):
        if cell.shape[0] != cell.shape[1] or cell.shape[0] != cube_length:
            continue

        if debug:
            image_copy = cv.cvtColor(image, cv.COLOR_GRAY2BGR).copy()

            pad_size = 10

            image_copy = np.pad(image_copy, ((
                pad_size, pad_size), (pad_size, pad_size), (0, 0)), constant_values=0)

            for point in [(x, y), (x + cube_length, y + cube_length)]:
                cv.circle(
                    image_copy, (point[0] + pad_size, point[1] + pad_size), 3, (0, 0, 255), 5)

            show_image(image_copy, f"{method_name} Extracting at these points")

        white_pixels_percentage = cv.countNonZero(
            cell) / float(cell.shape[0] * cell.shape[1])

        if white_pixels_percentage < 0.05:
            if debug:
                logger.info(
                    f"{method_name} Adding `None`, {white_pixels_percentage}.")

                show_image(cell, f"{method_name} < 5% white pixels")

            result.append(None)

            continue

        contours, _ = cv.findContours(
            cell, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            if debug:
                logger.info(f"{method_name} Adding `None`.")

            result.append(None)

            continue

        contour = max(contours, key=cv.contourArea)

        if debug:
            img = cv.cvtColor(cell, cv.COLOR_GRAY2BGR)

            cv.drawContours(img, [contour], -1, (0, 0, 255), 1)

            show_image(img, f"{method_name} ROI contour")


        mask = np.zeros_like(cell, dtype="uint8")

        cv.drawContours(
            image=mask,
            contours=[contour],
            contourIdx=-1,
            color=255,
            thickness=-1
        )

        digit = cv.bitwise_or(
            src1=cell,
            src2=cell,
            mask=mask
        )

        x, y, w, h = cv.boundingRect(contour)

        if debug:
            img = cv.cvtColor(cell, cv.COLOR_GRAY2BGR)

            cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)

            show_image(img, f"{method_name} Bounding rect. over ROI")


        ROI = digit[y: y + h, x: x + w]

        if debug:
            show_image(ROI, f"{method_name} ROI")


        if face.axis == Axis.Z:
            ROI = rotate_bound(ROI, -45)

            contours, _ = cv.findContours(
                image=ROI,
                mode=cv.RETR_EXTERNAL,
                method=cv.CHAIN_APPROX_SIMPLE
            )

            contour = max(contours, key=cv.contourArea)

            x, y, w, h = cv.boundingRect(contour)

            ROI = ROI[y: y + h, x: x + w]

            if debug:
                show_image(ROI, f"{method_name} ROI after rotation")

        result.append(cv.resize(ROI, constants.digit_size))

    return result
