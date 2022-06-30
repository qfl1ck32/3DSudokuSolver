"""
    Inspired from https://github.com/eborboihuc/rotate_3d
"""

import numpy as np
import cv2 as cv


class ImagePerspectiveTransformer:
    def __init__(self, image):
        self.image = image.copy()

        self.height = self.image.shape[0]
        self.width = self.image.shape[1]

        self.num_channels = 1 if len(
            self.image.shape) == 2 else self.image.shape[2]

    def rotate(self, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0):
        theta_radians, phi_radians, gamma_radians = list(map(np.deg2rad, [theta, phi, gamma]))

        d = np.sqrt(self.height ** 2 + self.width ** 2)

        self.focal = d / (2 * np.sin(gamma_radians) if np.sin(gamma_radians) != 0 else 1)

        dz = self.focal

        M = self.get_M(theta_radians, phi_radians, gamma_radians, dx, dy, dz)

        return cv.warpPerspective(self.image, M, (self.width, self.height))

    def get_M(self, theta: float, phi: float, gamma: float, dx: float, dy: float, dz: float):
        w = self.width
        h = self.height
        f = self.focal

        A1 = np.array([[1, 0, -w / 2],
                       [0, 1, -h / 2],
                       [0, 0, 1],
                       [0, 0, 1]])

        RX = np.array([[1, 0, 0, 0],
                       [0, np.cos(theta), -np.sin(theta), 0],
                       [0, np.sin(theta), np.cos(theta), 0],
                       [0, 0, 0, 1]])

        RY = np.array([[np.cos(phi), 0, -np.sin(phi), 0],
                       [0, 1, 0, 0],
                       [np.sin(phi), 0, np.cos(phi), 0],
                       [0, 0, 0, 1]])

        RZ = np.array([[np.cos(gamma), -np.sin(gamma), 0, 0],
                       [np.sin(gamma), np.cos(gamma), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])

        R = np.dot(np.dot(RX, RY), RZ)

        T = np.array([[1, 0, 0, dx],
                      [0, 1, 0, dy],
                      [0, 0, 1, dz],
                      [0, 0, 0, 1]])

        A2 = np.array([[f, 0, w/2, 0],
                       [0, f, h/2, 0],
                       [0, 0, 1, 0]])

        return np.dot(A2, np.dot(T, np.dot(R, A1)))
