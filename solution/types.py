import numpy as np
import cv2 as cv

from solution.enums import Axis, PathDirection

from typing import List, Union
from collections import defaultdict


from solution.exceptions import ThePathsCouldNotHaveBeenProperlyIdentified
from solution.globals import constants

class SudokuFace:
    coordinates: np.array
    digits: Union[np.ndarray, None]

    axis: Axis
    image: np.ndarray
    original_image: np.ndarray

    def __init__(self, coordinates: np.array, axis: Axis, image: np.ndarray):
        if image is None:
            raise ThePathsCouldNotHaveBeenProperlyIdentified()

        self.original_image = image

        self.image = cv.resize(image, constants.face_image_size)

        self.coordinates = coordinates
        self.axis = axis


        self.digits = None

    def set_digits(self, digits: np.ndarray):
        self.digits = digits


class SudokuPath:
    direction: PathDirection

    faces: List[SudokuFace]

    def __init__(self, faces: List[SudokuFace] = None):
        if faces is None:
            faces = []

        self.faces = faces

    def add_face(self, face: SudokuFace):
        self.faces.append(face)

    def set_direction(self, direction: PathDirection):
        self.direction = direction


PathMap = defaultdict[tuple, List[SudokuPath]]
