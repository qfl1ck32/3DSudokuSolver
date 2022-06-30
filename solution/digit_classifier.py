import os
import pickle
import cv2 as cv

import numpy.typing as npt

from packages.imaging import read_image

from solution.globals import constants

def _generate_digit_to_image_map():
    classes = [i for i in range(1, 10, 1)]

    mp = {k: [] for k in [i for i in range(1, 10, 1)]}

    for cls in classes:
        image = read_image(f"{constants.pattern_matching_digits_path}/{cls}.png")

        mp[cls] = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    return mp

class DigitClassifier:
    def __init__(self):
        if os.path.exists(constants.digit_to_image_map_file_path):
            with open(constants.digit_to_image_map_file_path, "rb") as file:
                self.digit_to_image_map = pickle.load(file)
        else:
            self.digit_to_image_map = _generate_digit_to_image_map()

            with open(constants.digit_to_image_map_file_path, "wb") as file:
                pickle.dump(self.digit_to_image_map, file)
    
    def classify(self, image: npt.NDArray):
        digit_to_score_map = {k: 0 for k in [i for i in range(1, 10, 1)]}

        for i in self.digit_to_image_map.keys():
            max_probability = 0

            current_template = self.digit_to_image_map[i]

            probability = cv.matchTemplate(
                image=image,
                templ=current_template,
                method=cv.TM_CCOEFF_NORMED
            )[0][0]

            if probability > max_probability:
                max_probability = probability

            digit_to_score_map[i] = max_probability


        best_digit = max(digit_to_score_map, key=digit_to_score_map.get)

        if digit_to_score_map[best_digit] < constants.digit_to_image_pattern_matching_threshold:
            return constants.face_empty_cell_value

        return best_digit
