import pickle
from timeit import default_timer
from typing import Callable, Dict, List

import cv2 as cv
import imutils
import numpy as np
from packages.imaging import four_point_transform, read_image
from packages.imaging import add_text_at_point, create_structure, extract_coordinates_from_contour, order_points, rotate_bound, unpad_image
from packages.imaging import show_image
from packages.logging import logger
from packages.logging import StepLogger
from packages.utility import get_current_function_name
from solution.digit_classifier import DigitClassifier

from solution.add_padding_to_coordinates import add_padding_to_coordinates
from solution.calculate_displacement_vectors import calculate_displacement_vectors
from solution.extract_face_digits import extract_face_digits

from solution.globals import constants, hyperparameters

from packages.geometry import calculate_iou
from solution.enums import Axis, SegmentType, PathDirection
from solution.exceptions import \
    check_thick_lines_points_have_been_identified_correctly
from solution.extract_face_if_exists import extract_face_if_exists
from solution.extract_starting_points_from_segments import extract_starting_points_from_segments
from solution.keep_edges_in_image import keep_edges_in_image
from solution.types import SudokuFace, SudokuPath

from packages.debugging import add_circle, add_circles_for_path

_extract_digits_for_faces_debug_index = 0


class SudokuCV:
    original_contour_image: np.ndarray
    contour_image: np.ndarray

    A: np.array
    B: np.array
    C: np.array
    D: np.array
    E: np.array
    F: np.array
    G: np.array

    vertical_segment: np.ndarray
    slash_segment: np.ndarray
    backslash_segment: np.ndarray

    transformation_end: np.ndarray

    vertical_segments_left_points: np.ndarray
    vertical_segments_right_points: np.ndarray

    backslash_segments_right_points: np.ndarray
    backslash_segments_down_points: np.ndarray

    slash_segments_up_points: np.ndarray
    slash_segments_down_points: np.ndarray

    horizontal_displacement_x: np.array
    vertical_displacement_x: np.array

    horizontal_displacement_y: np.array
    vertical_displacement_y: np.array

    horizontal_displacement_z: np.array
    vertical_displacement_z: np.array

    paths: List[SudokuPath]

    current_path: SudokuPath

    coordinates_to_face_map: Dict[tuple, SudokuFace]

    faces_coordinates: List[tuple]

    verbose: bool

    step_logger: StepLogger

    digit_classifier: DigitClassifier

    def __init__(self, image: np.ndarray,  verbose=False):
        self.step_logger = StepLogger(steps=8, verbose=verbose)

        self.contour_image = imutils.resize(
            image, width=constants.contour_image_width)

        self.original_contour_image = self.contour_image.copy()

        self.contour_image = cv.cvtColor(self.contour_image, cv.COLOR_BGR2GRAY)

        self.current_path = SudokuPath()

        self.paths = []

        self.coordinates_to_face_map = dict()
        self.faces_coordinates = []

        self.digit_classifier = DigitClassifier()

        self.verbose = verbose

        self.found_faces_coordinates_set = set()

    def log(self, message: str):
        if self.verbose:
            logger.info(message)

    def setup(self):
        debug = False

        if debug:
            method_name = get_current_function_name()

        #

        height, width = self.contour_image.shape[:2]

        self.A = np.array([0, int(height / 4)])
        self.B = np.array([int(width / 2), 0])
        self.C = np.array([width, int(height / 4)])
        self.D = np.array([width, int(3 * height / 4)])  # sad D
        self.E = np.array([int(width / 2), height])
        self.F = np.array([0, int(3 * height / 4)])
        self.G = np.array([int(width / 2), int(height / 2)])

        self.horizontal_displacement_x, self.vertical_displacement_x = calculate_displacement_vectors(
            [self.A, self.G, self.F])
        self.horizontal_displacement_y, self.vertical_displacement_y = calculate_displacement_vectors(
            [self.G, self.C, self.E])
        self.horizontal_displacement_z, self.vertical_displacement_z = calculate_displacement_vectors(
            [self.A, self.B, self.G])

        if debug:
            contour_image_copy = self.original_contour_image.copy()

            pad_size = 50

            contour_image_copy = np.pad(contour_image_copy, ((
                pad_size, pad_size), (pad_size, pad_size), (0, 0)), constant_values=255)

            for index, point in enumerate([self.A, self.B, self.C, self.D, self.E, self.F, self.G]):
                point = point + pad_size

                add_text_at_point(contour_image_copy, chr(
                    ord("A") + index), point, circle_size=20, text_scale=1, text_thickness=1)

            show_image(contour_image_copy, f"{method_name} Corner points")

        self.transformation_end = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype="float32")

    def extract_segments(self):
        debug = False

        if debug:
            method_name = get_current_function_name()

        #

        self.step_logger.log("Extracting the segments")

        height = self.contour_image.shape[0]

        pad_size = 10

        padded_contour_image = np.pad(self.contour_image, (
            (pad_size, pad_size),
            (pad_size, pad_size)),
            'constant',
            constant_values=255)

        vertical_segment_height = int(height / 6)

        vertical_segment_width = int(
            constants.segment_width_factor * vertical_segment_height)

        self.vertical_segment = create_structure(
            (int(vertical_segment_width * hyperparameters.struct_sizes_err), int(vertical_segment_height * hyperparameters.struct_sizes_err)))

        self.vertical_segment = np.pad(
            self.vertical_segment, ((20, 20), (20, 20)), constant_values=0)

        vertical_segment_of_real_size = create_structure(
            (vertical_segment_width, vertical_segment_height))

        # Slash
        slash_segments_image, angle, self.slash_segment = keep_edges_in_image(
            padded_contour_image, self.vertical_segment, [62, 61, 60, 59, 58])

        slash_segment_of_real_sizes = rotate_bound(
            vertical_segment_of_real_size, angle)
        slash_segment_of_real_sizes[slash_segment_of_real_sizes != 255] = 0

        slash_segments_image = unpad_image(slash_segments_image, pad_size)

        if debug:
            slash_segments_debug_image = cv.addWeighted(self.original_contour_image, 0.4,
                                                        slash_segments_image, 0.6, 1)
            show_image(slash_segments_debug_image,
                       f"{method_name} Slash segments")

        self.slash_segments_down_points, self.slash_segments_up_points = extract_starting_points_from_segments(
            slash_segments_image, SegmentType.SLASH, slash_segment_of_real_sizes, vertical_segment_of_real_size.shape)

        if debug:
            for point in self.slash_segments_down_points + self.slash_segments_up_points:
                add_circle(slash_segments_debug_image,
                           point, radius=2, thickness=8)

            show_image(slash_segments_debug_image,
                       f"{method_name} Slash segments with points")

        # Vertical
        vertical_segments_image, angle, self.vertical_segment = keep_edges_in_image(
            padded_contour_image, self.vertical_segment, [0])

        vertical_segments_image = unpad_image(
            vertical_segments_image, pad_size)

        if debug:
            vertical_segments_debug_image = cv.addWeighted(self.original_contour_image, 0.4,
                                                           vertical_segments_image, 0.6, 1)

            show_image(vertical_segments_debug_image,
                       f"{method_name} Vertical segments")

        vertical_struct_original_sizes = rotate_bound(
            vertical_segment_of_real_size, angle)
        vertical_struct_original_sizes[vertical_struct_original_sizes != 255] = 0

        self.vertical_segments_right_points, self.vertical_segments_left_points = extract_starting_points_from_segments(
            vertical_segments_image, SegmentType.VERTICAL, vertical_struct_original_sizes)

        if debug:
            for point in self.vertical_segments_left_points + self.vertical_segments_right_points:
                add_circle(vertical_segments_debug_image,
                           point, radius=2, thickness=8)

            show_image(vertical_segments_debug_image,
                       f"{method_name} Vertical segments with points")

        backslash_segments_image, angle, self.backslash_segment = keep_edges_in_image(
            padded_contour_image, self.vertical_segment, [-58, -59, -60, -61, -62])

        backslash_segment_of_real_sizes = rotate_bound(
            vertical_segment_of_real_size, angle)
        backslash_segment_of_real_sizes[backslash_segment_of_real_sizes != 255] = 0

        backslash_segments_image = unpad_image(
            backslash_segments_image, pad_size)

        if debug:
            backslash_segments_debug_image = cv.addWeighted(self.original_contour_image, 0.4,
                                                            backslash_segments_image, 0.6, 1)

            show_image(backslash_segments_debug_image,
                       f"{method_name} Backslash segments")

        self.backslash_segments_right_points, self.backslash_segments_down_points = extract_starting_points_from_segments(
            backslash_segments_image, SegmentType.BACKSLASH, backslash_segment_of_real_sizes, vertical_segment_of_real_size.shape)

        if debug:
            for point in self.backslash_segments_down_points + self.backslash_segments_right_points:
                add_circle(backslash_segments_debug_image,
                           point, radius=2, thickness=8)

            show_image(backslash_segments_debug_image,
                       f"{method_name} Backslash segments with points")

        check_thick_lines_points_have_been_identified_correctly(
            self.vertical_segments_left_points,
            self.vertical_segments_right_points,
            self.backslash_segments_right_points,
            self.backslash_segments_down_points,
            self.slash_segments_up_points,
            self.slash_segments_down_points
        )

        if debug:
            points = []

            all_segments_image = backslash_segments_image & slash_segments_image & vertical_segments_image

            all_segments_and_points = cv.addWeighted(self.original_contour_image,
                                                     0.2, all_segments_image, 0.8, 1)

            points.extend(self.vertical_segments_left_points)
            points.extend(self.vertical_segments_right_points)
            points.extend(self.backslash_segments_down_points)
            points.extend(self.backslash_segments_right_points)
            points.extend(self.slash_segments_down_points)
            points.extend(self.slash_segments_up_points)

            show_image(all_segments_and_points,
                       f"{method_name} All edges and points")

            for point in points:
                add_circle(all_segments_and_points,
                           point, radius=2, thickness=4)

            show_image(all_segments_and_points,
                       f"{method_name} All edges and points")

    def extract_paths(self):
        self.step_logger.log("Extracting vertical right paths")

        # Vertical
        for a in self.vertical_segments_right_points:
            self.extract_path(self.extract_vertical_right_path_get_first_coordinates, Axis.Y,
                              self.extract_vertical_right_path_get_second_coordinates, Axis.X, self.extract_vertical_right_path_get_next_points, a)

            self.store_current_path(
                PathDirection.HORIZONTAL)

        self.step_logger.log("Extracting vertical left paths")

        for b in self.vertical_segments_left_points:
            self.extract_path(self.extract_vertical_left_path_get_first_coordinates, Axis.Y,
                              self.extract_vertical_left_path_get_second_coordinates, Axis.X, self.extract_vertical_left_path_get_next_point, b)

            self.store_current_path(
                PathDirection.HORIZONTAL)

        # Slash
        self.step_logger.log("Extracting slash up paths")
        for a in self.slash_segments_up_points:
            self.extract_path(self.extract_slash_up_path_get_first_coordinates, Axis.Y,
                              self.extract_slash_up_path_get_second_coordinates, Axis.Z, self.extract_slash_up_path_get_next_point, a)

            self.store_current_path(PathDirection.VERTICAL)

        self.step_logger.log("Extracting slash down paths")

        for a in self.slash_segments_down_points:
            self.extract_path(self.extract_slash_down_path_get_first_coordinates, Axis.Y,
                              self.extract_slash_down_path_get_second_coordinates, Axis.Z, self.extract_slash_down_path_get_next_point, a)

            self.store_current_path(PathDirection.VERTICAL)

        self.step_logger.log("Extracting backslash right paths")

        # Backslash
        for a in self.backslash_segments_right_points:
            self.extract_path(self.extract_backslash_right_path_get_first_coordinates, Axis.Z,
                              self.extract_backslash_right_path_get_second_coordinates, Axis.X, self.extract_backslash_right_path_get_next_point, a)

            is_only_x = all(
                face.axis == Axis.X for face in self.current_path.faces)

            direction = PathDirection.HORIZONTAL

            if is_only_x:
                direction = PathDirection.VERTICAL

            self.store_current_path(direction)

        self.step_logger.log("Extracting backslash down paths")

        for b in self.backslash_segments_down_points:
            self.extract_path(self.extract_backslash_down_path_get_first_coordinates, Axis.Z,
                              self.extract_backslash_down_path_get_second_coordinates, Axis.X, self.extract_backslash_down_path_get_next_point, b)

            direction = PathDirection.VERTICAL

            is_only_z = all(
                face.axis == Axis.Z for face in self.current_path.faces)

            if is_only_z:
                direction = PathDirection.HORIZONTAL

            self.store_current_path(direction)

    def extract_digits_for_faces(self):
        global _extract_digits_for_faces_debug_index

        method_name = f"{get_current_function_name()} - {_extract_digits_for_faces_debug_index}"

        #

        self.step_logger.log("Extracting digits")

        for face in self.coordinates_to_face_map.values():
            _extract_digits_for_faces_debug_index += 1

            if face.digits is not None:
                continue

            digits = self.extract_numbers_from_face(face)
            face.set_digits(digits)

            debug = False

            if debug:
                print(digits)
                show_image(
                    face.image, f"{method_name} Current face [{_extract_digits_for_faces_debug_index}]")

    def extract_path(self, get_coordinates_1: Callable, axis_1: Axis, get_coordinates_2: Callable, axis_2: Axis, get_next_point: Callable, current_point: np.array):
        if self.should_stop_path_extraction_recursion():
            return

        coordinates = get_coordinates_1(current_point)

        axis = axis_1

        face_image, exact_coordinates = self.extract_face_given_coordinates(
            coordinates, axis)

        if face_image is None:
            axis = axis_2
            coordinates = get_coordinates_2(current_point)

            face_image, exact_coordinates = self.extract_face_given_coordinates(
                coordinates, axis)

        self.add_face_to_current_path(
            SudokuFace(exact_coordinates, axis, face_image))

        return self.extract_path(get_coordinates_1, axis_1, get_coordinates_2, axis_2, get_next_point, get_next_point(coordinates, axis))

    # Slash down
    def extract_slash_down_path_get_first_coordinates(self, a: np.array):
        b = a - self.horizontal_displacement_y
        c = a - self.vertical_displacement_y - self.horizontal_displacement_y
        d = a - self.vertical_displacement_y

        return (a, b, c, d)

    def extract_slash_down_path_get_second_coordinates(self, a: np.array):
        b = a - self.horizontal_displacement_z
        c = a - self.vertical_displacement_z - self.horizontal_displacement_z
        d = a - self.vertical_displacement_z

        return (a, b, c, d)

    def extract_slash_down_path_get_next_point(self, coordinates: np.array, _):
        (_, _, _, d) = coordinates

        return d

    # Slash up
    def extract_slash_up_path_get_first_coordinates(self, d: np.array):
        a = d + self.vertical_displacement_y
        b = a - self.horizontal_displacement_y
        c = a - self.horizontal_displacement_y - self.vertical_displacement_y

        return (a, b, c, d)

    def extract_slash_up_path_get_second_coordinates(self, d: np.array):
        a = d + self.vertical_displacement_z
        b = a - self.horizontal_displacement_z
        c = a - self.vertical_displacement_z - self.horizontal_displacement_z

        return (a, b, c, d)

    def extract_slash_up_path_get_next_point(self, coordinates: np.array, _):
        (a, _, _, _) = coordinates

        return a

    # Vertical right
    def extract_vertical_right_path_get_first_coordinates(self, a: np.array):
        b = a - self.horizontal_displacement_y
        c = a - self.vertical_displacement_y - self.horizontal_displacement_y
        d = a - self.vertical_displacement_y

        return (a, b, c, d)

    def extract_vertical_right_path_get_second_coordinates(self, a: np.array):
        b = a - self.horizontal_displacement_x
        c = a - self.horizontal_displacement_x - self.vertical_displacement_x
        d = a - self.vertical_displacement_x

        return (a, b, c, d)

    def extract_vertical_right_path_get_next_points(self, coordinates: np.array, _):
        (_, b, _, _) = coordinates

        return b

    # Vertical left

    def extract_vertical_left_path_get_first_coordinates(self, b: np.array):
        a = b + self.horizontal_displacement_y
        c = a - self.vertical_displacement_y - self.horizontal_displacement_y
        d = a - self.vertical_displacement_y

        return (a, b, c, d)

    def extract_vertical_left_path_get_second_coordinates(self, b: np.array):
        a = b + self.horizontal_displacement_x
        c = a - self.horizontal_displacement_x - self.vertical_displacement_x
        d = a - self.vertical_displacement_x

        return (a, b, c, d)

    def extract_vertical_left_path_get_next_point(self, coordinates: np.array, _):
        (a, _, _, _) = coordinates

        return a

    # Backslash right
    def extract_backslash_right_path_get_first_coordinates(self, a: np.array):
        b = a - self.horizontal_displacement_z
        c = a - self.vertical_displacement_z - self.horizontal_displacement_z
        d = a - self.vertical_displacement_z

        return (a, b, c, d)

    def extract_backslash_right_path_get_second_coordinates(self, d: np.array):
        a = d + self.vertical_displacement_x
        b = a - self.horizontal_displacement_x
        c = a - self.horizontal_displacement_x - self.vertical_displacement_x

        return (a, b, c, d)

    def extract_backslash_right_path_get_next_point(self, coordinates: np.array, face: Axis):
        (a, b, _, _) = coordinates

        return b if face == Axis.Z else a

    # Backslash left
    def extract_backslash_down_path_get_first_coordinates(self, b: np.array):
        a = b + self.horizontal_displacement_z
        c = a - self.vertical_displacement_z - self.horizontal_displacement_z
        d = a - self.vertical_displacement_z

        return (a, b, c, d)

    def extract_backslash_down_path_get_second_coordinates(self, a: np.array):
        b = a - self.horizontal_displacement_x
        c = a - self.vertical_displacement_x - self.horizontal_displacement_x
        d = a - self.vertical_displacement_x

        return (a, b, c, d)

    def extract_backslash_down_path_get_next_point(self, coordinates: np.array, face: Axis):
        (a, _, _, d) = coordinates

        return a if face == Axis.Z else d

    # Next stuff
    def extract_face_given_coordinates(self, coordinates: tuple, axis: Axis):
        debug = False

        if debug:
            method_name = get_current_function_name()

        #

        coordinates = add_padding_to_coordinates(np.array(coordinates), axis)

        if debug:
            self.log(f"{method_name} Axis: {axis}")

            contour_image_copy = self.original_contour_image.copy()

            pad_size = 50

            contour_image_copy = np.pad(contour_image_copy, ((
                pad_size, pad_size), (pad_size, pad_size), (0, 0)), constant_values=255)

            for point in coordinates:
                cv.circle(contour_image_copy, (int(
                    point[0]) + pad_size, int(point[1]) + pad_size), 3, (0, 0, 255), 15)

            show_image(
                contour_image_copy, f"{method_name} Extracting here")

        # TODO: this optimisation should work. In reality it isn't really an optimisation, sadly, though.
        # for cached_coordinates in self.coordinates_to_face_map.keys():
        #     if calculate_iou(cached_coordinates, coordinates) > hyperparameters.iou_threshold:
        #         face = self.coordinates_to_face_map[cached_coordinates]

        #         if debug:
        #             self.log(
        #                 f"{method_name} Got cached version for {coordinates}, at {cached_coordinates}")

        #             show_image(face.image, f"{method_name} Cached face")

        #         return face.image, face.coordinates

        # if len(self.coordinates_to_faces_map.keys()) == 27:
        #     if debug:
        #         self.log(
        #             f"{method_name} All the faces have been identified, and the given coordinates do not form a face.")
        #     return None

        face_image, transformation_matrix = four_point_transform(
            self.original_contour_image, coordinates)

        if debug:
            show_image(face_image, f"{method_name} Warped image")

        new_face_image, coordinates_correction = extract_face_if_exists(
            face_image)

        true_coordinates = coordinates

        if new_face_image is not None:
            a, b, c, d = coordinates_correction

            M_inverse = np.linalg.inv(transformation_matrix)

            points = np.float32([[a], [b], [c], [d]])

            perspective_transformed_points = cv.perspectiveTransform(
                points, M_inverse)

            true_coordinates = perspective_transformed_points[:, 0]

        if debug:
            if new_face_image is not None:
                show_image(new_face_image, f"{method_name} The face exists")

            else:
                show_image(face_image,
                           f"{method_name} The face does not exist")

        # TODO: a bit ugly :D
        true_coordinates = tuple(tuple((int(coordinates[0]), int(
            coordinates[1]))) for coordinates in true_coordinates)

        return new_face_image, true_coordinates

    def add_face_to_current_path(self, face: SudokuFace):
        found_cached = False

        face_coordinates = face.coordinates

        for coordinates in self.coordinates_to_face_map.keys():
            iou = calculate_iou(face_coordinates, coordinates)

            if iou > constants.iou_threshold:
                face = self.coordinates_to_face_map[coordinates]
                found_cached = True

                break

        if not found_cached:
            self.coordinates_to_face_map[face_coordinates] = face

        self.current_path.add_face(face)

    def should_stop_path_extraction_recursion(self):
        return len(self.current_path.faces) == constants.cube_size

    def store_current_path(self, direction: PathDirection):
        self.current_path.set_direction(direction)

        self.paths.append(self.current_path)

        self.current_path = SudokuPath()

    def extract_numbers_from_face(self, face: SudokuFace):
        digit_images = extract_face_digits(face)

        answer = np.zeros(
            (constants.cube_size, constants.cube_size), dtype=np.int32)

        for row in range(constants.cube_size):
            for column in range(constants.cube_size):
                digit_image = digit_images[constants.cube_size * row + column]

                if digit_image is not None:
                    answer[row][column] = self.digit_classifier.classify(
                        digit_image)

        return answer

    def save_paths(self, path=constants.paths_file_path):
        with open(path, "wb") as file:
            pickle.dump(self.paths, file)

    def load_paths(self, path=constants.paths_file_path) -> List[SudokuPath]:
        with open(path, "rb") as file:
            self.paths = pickle.load(file)

    def add_numbers_on_face(self, face: SudokuFace):
        image = face.original_image.copy()

        face_image = cv.cvtColor(face.original_image, cv.COLOR_BGR2GRAY)

        face_image = cv.threshold(face_image, 250, 255, cv.THRESH_BINARY)[1]

        face_image = cv.dilate(face_image, np.ones((2, 2)), iterations=2)

        contours = cv.findContours(
            face_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]

        for contour in contours:
            corners = order_points(extract_coordinates_from_contour(contour))

            top_left_corner, bottom_right_corner = corners[0], corners[2]

            width, height = bottom_right_corner[1] - \
                top_left_corner[1], bottom_right_corner[0] - top_left_corner[0]

            image[top_left_corner[1]: top_left_corner[1] + width,
                  top_left_corner[0]: top_left_corner[0] + height] = 255

        for row in range(constants.cube_size):
            for column in range(constants.cube_size):
                number = face.digits[row][column]

                if number == constants.face_empty_cell_value:
                    continue

                (width, height) = image.shape[:2]

                epsilon = -2

                one_cube_size = int(width / 3)

                image = cv.resize(image, (width, width))

                # TODO: could optimise
                digit_image = read_image(
                    f"{constants.rotated_digits_path if face.axis == Axis.Z else constants.digits_path}/{number}.jpg")

                digit_image = cv.bitwise_not(digit_image)

                digit_image = cv.resize(
                    digit_image, (one_cube_size, one_cube_size))

                digit_image = np.bitwise_not(digit_image)

                cpy = cv.cvtColor(digit_image, cv.COLOR_BGR2GRAY)
                thresh = cv.threshold(cpy, 127, 255, cv.THRESH_BINARY_INV)[1]

                contours = cv.findContours(
                    thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]

                contour = max(contours, key=cv.contourArea)

                x, y, w, h = cv.boundingRect(contour)

                pad_size = 5

                x -= pad_size
                y -= pad_size
                w += 2 * pad_size
                h += 2 * pad_size

                y, x, w, h = int(row * one_cube_size), int(column *
                                                           one_cube_size), one_cube_size, one_cube_size

                center = (2 * x + w) // 2 - epsilon, (2 * y + h) // 2

                digit_size = one_cube_size // 3

                top_left_corner = center[0] - \
                    digit_size, center[1] - digit_size
                bottom_right_corner = center[0] + \
                    digit_size, center[1] + digit_size

                size = bottom_right_corner[0] - top_left_corner[0]

                digit_image = cv.resize(
                    digit_image, (size, size), interpolation=cv.INTER_AREA)

                image[top_left_corner[1]: top_left_corner[1] + size,
                      top_left_corner[0]: top_left_corner[0] + size] = digit_image

        image = cv.resize(image, (height, width))

        return image

    def add_face_on_image(self, face: SudokuFace, contour_image: np.ndarray):
        a, b, c, d = face.coordinates

        face_image = self.add_numbers_on_face(face)

        (width, height) = contour_image.shape[:2]

        face_image = cv.resize(face_image, (height, width))

        transformation = cv.getPerspectiveTransform(
            self.transformation_end, np.array([a, b, c, d], dtype="float32"))

        transformed_face_image = cv.warpPerspective(
            face_image, transformation, (height, width))

        mask = np.zeros(transformed_face_image.shape[:2], dtype="uint8")

        contours, _ = cv.findContours(
            cv.cvtColor(transformed_face_image, cv.COLOR_BGR2GRAY), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        cv.drawContours(mask, contours, -1, 255, -1)

        masked_contour_image = cv.bitwise_and(
            contour_image, contour_image, mask=~mask)

        return masked_contour_image | transformed_face_image

    def generate_solution_image(self, faces):
        image = self.original_contour_image.copy()

        for face in faces:
            image = self.add_face_on_image(face, image)

        return image
