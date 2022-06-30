import cv2 as cv
import numpy as np

from solution.types import SudokuPath


def add_circle(image: np.ndarray, center: tuple, color=(0, 0, 255), radius=3, thickness=20):
    cv.circle(image, center, radius, color, thickness)

def add_circles_for_path(path: SudokuPath, image: np.ndarray, pad_size=0):
    if pad_size != 0:
        img = np.pad(image, ((pad_size, pad_size), (pad_size,
                                                    pad_size), (0, 0)), constant_values=255)
    else:
        img = image.copy()

    for face in path.faces:
        for point in face.coordinates:
            cv.circle(img, (point[0] + pad_size,
                      point[1] + pad_size), 5, (0, 0, 255), 10)

    return img
