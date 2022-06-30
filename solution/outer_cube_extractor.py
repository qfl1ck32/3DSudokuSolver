import cv2 as cv
import numpy as np
from packages.imaging import read_image

from solution.image_transformer import ImagePerspectiveTransformer

from solution.globals import constants

from packages.imaging import show_image

class OuterCubeExtractor:
    def __init__(self):
        self.contour_template_image = read_image(
            constants.pattern_matching_cube_countour_image_path)

        self.template_size = (400, 400)

        self.contour_template_image = cv.resize(self.contour_template_image, self.template_size)
        self.contour_template_image = cv.cvtColor(self.contour_template_image, cv.COLOR_BGR2GRAY)
        self.contour_template_image[self.contour_template_image < 127] = 0
        self.contour_template_image[self.contour_template_image >= 127] = 255

        self.contour_template_image_number_of_white_pixels = np.sum(
            self.contour_template_image == 255)


    def extract(self, image: np.ndarray):
        best_top = None
        best_left = None
        best_right = None
        best_bottom = None

        best_theta = 0
        best_phi = 0
        best_gamma = 0

        best_match = 0

        grayscale = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

        blurred = cv.GaussianBlur(grayscale,
                                ksize=(3, 3),
                                sigmaX=3)

        thresholded = cv.adaptiveThreshold(blurred,
                                        maxValue=255,
                                        adaptiveMethod=cv.ADAPTIVE_THRESH_MEAN_C,
                                        thresholdType=cv.THRESH_BINARY,
                                        blockSize=25,
                                        C=2)

        negated = cv.bitwise_not(thresholded)

        contours, _ = cv.findContours(negated,
                                    mode=cv.RETR_EXTERNAL,
                                    method=cv.CHAIN_APPROX_SIMPLE)

        contour = max(contours, key=cv.contourArea)

        mask = np.zeros(shape=image.shape[:2], dtype=np.uint8)

        contour_image = cv.drawContours(image=mask,
                                        contours=[contour],
                                        contourIdx=-1,
                                        color=255,
                                        thickness=-1)

        c = cv.cvtColor(contour_image, cv.COLOR_GRAY2BGR) & image

        area = cv.countNonZero(contour_image) / \
            float(contour_image.shape[0] * contour_image.shape[1])

        for theta in np.arange(-2, 2.1, 0.5):
            theta_image = ImagePerspectiveTransformer(
                contour_image).rotate(theta=theta)
            for gamma in np.arange(-2, 2.1, 0.5):
                gamma_image = ImagePerspectiveTransformer(
                    theta_image).rotate(gamma=gamma)
                for phi in np.arange(-2, 2.1, 0.5):
                    current_image = ImagePerspectiveTransformer(
                        gamma_image).rotate(phi=phi)

                    contours = cv.findContours(image=current_image,
                                            mode=cv.RETR_EXTERNAL,
                                            method=cv.CHAIN_APPROX_SIMPLE)[0]

                    contour = max(contours, key=cv.contourArea)

                    top = contour[contour[:, :, 1].argmin()][0]
                    right = contour[contour[:, :, 0].argmax()][0]
                    bottom = contour[contour[:, :, 1].argmax()][0]
                    left = contour[contour[:, :, 0].argmin()][0]

                    area = cv.countNonZero(
                        current_image) / float(current_image.shape[1] * current_image.shape[0])

                    if area > 0.6:
                        continue

                    contours, _ = cv.findContours(image=current_image,
                                                mode=cv.RETR_EXTERNAL,
                                                method=cv.CHAIN_APPROX_SIMPLE)

                    contour = max(contours, key=cv.contourArea)

                    top = contour[contour[:, :, 1].argmin()][0]
                    right = contour[contour[:, :, 0].argmax()][0]
                    bottom = contour[contour[:, :, 1].argmax()][0]
                    left = contour[contour[:, :, 0].argmin()][0]

                    cube = current_image[top[1]: bottom[1], left[0]: right[0]]

                    cube = cv.resize(cube, self.template_size)

                    cube[cube < 127] = 2
                    cube[cube >= 127] = 255

                    match = np.sum(
                        cube == self.contour_template_image) / self.contour_template_image_number_of_white_pixels

                    if match > best_match:
                        best_match = match

                        best_top = top
                        best_right = right
                        best_bottom = bottom
                        best_left = left

                        best_theta = theta
                        best_gamma = gamma
                        best_phi = phi

                    if match > 0.99:
                        break

        image = ImagePerspectiveTransformer(image).rotate(
            best_theta, best_phi, best_gamma)

        cube = image[best_top[1]: best_bottom[1], best_left[0]: best_right[0]]

        cube_for_contour = cv.bitwise_not(cv.cvtColor(cube, cv.COLOR_BGR2GRAY))

        cube_for_contour[cube_for_contour < 196] = 0

        contours, _ = cv.findContours(
            cube_for_contour, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        contour = max(contours, key=cv.contourArea)

        top = contour[contour[:, :, 1].argmin()][0]
        right = contour[contour[:, :, 0].argmax()][0]
        bottom = contour[contour[:, :, 1].argmax()][0]
        left = contour[contour[:, :, 0].argmin()][0]

        cube = cube[top[1]: bottom[1], left[0]: right[0]]

        return cube
