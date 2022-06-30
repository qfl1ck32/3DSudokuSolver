import numpy as np
import cv2 as cv

class Constants:
    def __init__(self):
        self.cube_size = 3
        self.cubed_cube_size = np.power(self.cube_size, 3)

        self.digit_size = (28, 28)
        self.segment_width_factor = 0.09

        self.face_image_size = (128, 128)

        self.contour_image_width = 640

        self.face_empty_cell_value = 0

        self.iou_threshold = 0.5
        
        self.digit_to_image_pattern_matching_threshold = 0.5

        self.data_path = "./data"

        self.extract_face_digits_element_horizontal = cv.getStructuringElement(
        cv.MORPH_RECT, (self.face_image_size[0] // 2, 1))
        self.extract_face_digits_element_vertical = cv.getStructuringElement(
        cv.MORPH_RECT, (1, self.face_image_size[1] // 2))

        self.extract_face_digits_element_noise = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))

        self.solution_path = f"{self.data_path}/solution"
        self.images_path = f"{self.data_path}/images"
        self.pattern_matching_path = f"{self.data_path}/pattern_matching"

        self.digits_path = f"{self.data_path}/digits"
        self.rotated_digits_path = f"{self.digits_path}/rotated_45"

        self.pattern_matching_cube_countour_image_path = f"{self.pattern_matching_path}/outer_cube_detection/contour.jpg"
        self.pattern_matching_digits_path = f"{self.pattern_matching_path}/digits"

        self.digit_to_image_map_file_path = f"{self.pattern_matching_digits_path}/digit_to_image_map.pkl"

        self.digits_files_path = f"{self.data_path}/digits/"
        self.sudoku_validation_files_path = f"{self.data_path}/sudoku_validation_files"

        self.paths_file_path = f"{self.solution_path}/paths.pkl"

        self.black_low = np.array([0, 0, 0])
        self.black_high = np.array([255, 255, 100])
