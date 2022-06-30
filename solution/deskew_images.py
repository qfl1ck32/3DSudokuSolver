from typing import List
import cv2
import os

from timeit import default_timer

from packages.imaging import read_image
from packages.logging import logger

from solution.globals import constants
from solution.outer_cube_extractor import OuterCubeExtractor


def deskew_images(file_names: List[str] = None):
    outer_cube_extractor = OuterCubeExtractor()

    if file_names is None:
        file_names = os.listdir(constants.images_path)

    for file_name in file_names:
        start_time = default_timer()

        logger.info(f"Deskewing {file_name}...")

        image = read_image(f"{constants.images_path}/{file_name}")

        cube = outer_cube_extractor.extract(image)

        [name, extension] = file_name.split(".")

        cv2.imwrite(f"{constants.images_path}/deskewed/{name}.{extension}", cube)

        logger.info(f"Done - {default_timer() - start_time}.\n")
