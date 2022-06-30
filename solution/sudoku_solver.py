from typing import List
import numpy as np


import cv2 as cv

from scipy.ndimage import rotate
from packages.debugging  import add_circles_for_path, add_circle
from packages.imaging import show_image
from packages.utility import get_current_function_name
from solution.sudoku_cv import SudokuCV
from packages.logging import logger
from solution.create_path_map import create_path_map

from solution.enums import Axis, PathDirection
from solution.types import PathMap, SudokuFace



class SudokuSolver:
    path_map: PathMap

    faces: List[SudokuFace]

    recursive_calls: int

    sudoku_cv: SudokuCV

    completed_image: np.ndarray

    _debug_number: int

    def __init__(self, sudoku_cv: SudokuCV):
        self.path_map = create_path_map(sudoku_cv.paths)

        self.sudoku_cv = sudoku_cv

        self.faces = []

        self._debug_number = float('+inf')

        self.faces = []

        """
        It might be tempting to directly use `sudoku_cv.coordinate_to_face_map.values()`, instead of iterating through the path map.
        The problem is that the order in which we solve the faces matters.
        
        Directly using coordinate_to_face_map.values() we're going to end up with all the faces but from unrelated paths.
            This way, the algorithm will find a solution for the three faces of a path,
            then for another path, and so on, until it reaches a path which contains a face that has been previously solved,
            and only at this point it finds out that the first choices were wrong. Going back to square one will take a lot of time.
        """
        for paths in self.path_map.values():
            for path in paths:
                for face in path.faces:
                    if face not in self.faces:
                        self.faces.append(face)

        self.completed_image = sudoku_cv.original_contour_image.copy()

        self.recursive_calls = 0

    def _is_debug(self):
        return self.recursive_calls > self._debug_number

    def _debug_is_move_legal(self, given_face: SudokuFace, row: int, column: int, number: int, answer: bool):
        if not self._is_debug():
            return

        cube_image = self.completed_image.copy()

        logger.info(f"{get_current_function_name()} Putting {number} on {row} / {column} => {answer}")

        paths = self.path_map[given_face.coordinates]

        for point in given_face.coordinates:
            add_circle(cube_image, point, radius=8, thickness=8)

        cube_image = add_circles_for_path(paths[0], cube_image)

        show_image(cube_image, f"{get_current_function_name()} This is the path")

    def is_state_valid(self, given_face: SudokuFace, row: int, column: int, number: int):
        if given_face.digits[row][column] != 0:
            self._debug_is_move_legal(given_face, row, column, number, False)
            return False

        if number in given_face.digits:
            self._debug_is_move_legal(given_face, row, column, number, False)
            return False

        paths = self.path_map[given_face.coordinates]

        for path in paths:
            all_axis = list(map(lambda f: f.axis, path.faces))

            has_x_z_perspective_change = Axis.X in all_axis and Axis.Z in all_axis

            direction = path.direction

            if has_x_z_perspective_change:
                if given_face.axis == Axis.X:
                    direction = PathDirection.VERTICAL
                elif given_face.axis == Axis.Z:
                    direction = PathDirection.HORIZONTAL

            for face in path.faces:
                digits = face.digits

                if has_x_z_perspective_change:
                    if given_face.axis == Axis.Z and face.axis == Axis.X:
                        digits = rotate(digits, -90)

                    elif given_face.axis == Axis.X and face.axis == Axis.Z:
                        digits = rotate(digits, 90)

                slice_to_verify = digits[:,
                                         column] if direction == PathDirection.VERTICAL else digits[row, :]

                if self._is_debug():
                    logger.info(f"{get_current_function_name()} Verifying for: \n{digits}")
                    logger.info(f"{get_current_function_name()} Extracting, for direction {direction}: {slice_to_verify}")

                if number in slice_to_verify:
                    self._debug_is_move_legal(
                        given_face, row, column, number, False)

                    return False

        self._debug_is_move_legal(given_face, row, column, number, True)

        return True

    def find_empty_face_and_coordinates(self):
        for face in self.faces:
            rows, cols = np.where(face.digits == 0)

            if len(rows) == 0:
                continue

            return face, rows[0], cols[0]

        return None

    def solve(self):
        self.recursive_calls += 1

        find_result = self.find_empty_face_and_coordinates()

        if find_result is None:
            return True
        else:
            face, row, column = find_result

        for number in range(1, 10):
            if self.is_state_valid(face, row, column, number):
                face.digits[row][column] = number

                if self._is_debug():
                    self.completed_image = self.sudoku_cv.add_face_on_image(
                        face, self.completed_image)

                if self.solve():
                    return True

                face.digits[row][column] = 0

                if self._is_debug():
                    self.completed_image = self.sudoku_cv.add_face_on_image(
                        face, self.completed_image)

        return False
