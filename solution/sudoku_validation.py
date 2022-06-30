import itertools
import os
import pickle
from timeit import default_timer

import numpy as np
from matplotlib import pyplot as plt
from packages.imaging import read_image
from packages.logging import logger
from tqdm import tqdm

from solution.globals import constants, hyperparameters
from solution.sudoku_cv import SudokuCV
from solution.sudoku_solver import SudokuSolver


class SudokuValidation:
    def __init__(self):
        pass

    def generate_validation_set(self, size=100):
        images = []

        for i in range(1, 326, 1):
            if len(images) == size:
                break

            x = np.random.random(1)[0]

            if x > 0.5:
                images.append(read_image(f"{constants.images_path}/deskewed/{i}.jpg"))

        return images

    def find_struct_size_and_keep_edges_in_image_threshold(self):
        struct_sizes_err = np.arange(0.5, 0.9, 0.1)
        keep_edges_in_image_threshold = np.arange(127, 250, 1)

        all_combinations = list(itertools.product(
            struct_sizes_err, keep_edges_in_image_threshold))

        max_err_count = 5
        best_err_count = max_err_count
        best_values = []

        validation_files = self.generate_validation_set()

        for values in tqdm(all_combinations):
            struct_size_err, keep_edges_in_image_threshold = values

            hyperparameters.struct_sizes_err = struct_size_err
            hyperparameters.keep_edges_in_image_threshold = keep_edges_in_image_threshold

            err_count = 0

            for image in validation_files:
                try:
                    SudokuCV(image, verbose=False).extract_segments()
                except Exception:
                    err_count += 1

                if err_count > max_err_count:
                    break

            if err_count < best_err_count:
                best_values = values
                best_err_count = err_count

            logger.info(best_err_count, best_values)

        logger.info(best_err_count)
        logger.info(best_values)

    def calculate_timing_statistics(self):
        if os.path.exists(f"{constants.solution_path}/timing.pkl"):
            with open(f"{constants.solution_path}/timing.pkl", "rb") as f:
                data = pickle.load(f)
        else:
            data = []

        for i in range(1 + len(data), 326, 1):
            logger.info(f"Solving Cube {i}")

            image = read_image(f"{constants.images_path}/deskewed/{i}.jpg")

            sudoku_cv = SudokuCV(image, False)

            sudoku_cv.setup()

            time_info = default_timer()

            sudoku_cv.extract_segments()

            sudoku_cv.extract_paths()

            sudoku_cv.extract_digits_for_faces()

            end_time_info = default_timer()

            solver = SudokuSolver(sudoku_cv)

            time_solve = default_timer()

            solved = solver.solve()

            end_time_solve = default_timer()

            logger.info(f"Solved! Solution has been found: {solved}")

            logger.info(solver.recursive_calls)

            data.append([end_time_info - time_info, end_time_solve - time_solve])

            with open(f"{constants.solution_path}/timing.pkl", "wb") as file:
                pickle.dump(data, file)
    
    def plot_timing(self):
        standard_items = 163

        with open(f"{constants.solution_path}/timing.pkl", "rb") as file:
            data = pickle.load(file)

        info_extraction_times = np.array(list(map(lambda c: c[0], data)))
        solution_times = np.array(list(map(lambda c: c[1], data)))

        mean_value_info = info_extraction_times.mean()
        mean_value_solution = solution_times.mean()

        info_standard = np.array(info_extraction_times.tolist()[:standard_items])
        info_challenging = np.array(info_extraction_times.tolist()[standard_items:])

        solution_times_standard = np.array(solution_times.tolist()[:standard_items])
        solution_times_challenging = np.array(solution_times.tolist()[standard_items:])

        full_times = info_extraction_times + solution_times
        full_times_standard  = info_standard + solution_times_standard
        full_times_challenging = info_challenging + solution_times_challenging

        x = 1 + np.arange(len(info_extraction_times))

        plt.figure(figsize=(12, 6), dpi=150)

        plt.plot(x, info_extraction_times)
        plt.ylabel("Timpul necesar extragerii informațiilor vizuale (s)")
        plt.xlabel("Numărul imaginii")

        plt.axhline(y=mean_value_info, color='r', linestyle='dashed', label='Media pe toate imaginile')
        plt.axhline(y=info_standard.mean(), color='orange', linestyle='dashed', label='Media pe imaginile de dificultate standard')
        plt.axhline(y=info_challenging.mean(), color='purple', linestyle='dashed', label='Media pe imaginile de dificultate ridicată')

        plt.axvline(x=163, color='g', label='Schimbarea nivelului de dificultate')

        plt.legend(bbox_to_anchor=(0.795, 1), loc='upper center')

        plt.show()

        #

        plt.figure(figsize=(12, 6), dpi=150)

        x = 1 + np.arange(len(data))

        plt.plot(x, solution_times)
        plt.ylabel("Timpul necesar generării soluției (s)")
        plt.xlabel("Numărul imaginii")

        plt.axhline(y=mean_value_solution, color='r', linestyle='dashed', label='Media pe toate imaginile')
        plt.axhline(y=solution_times_standard.mean(), color='orange', linestyle='dashed', label='Media pe imaginile de dificultate standard')
        plt.axhline(y=solution_times_challenging.mean(), color='purple', linestyle='dashed', label='Media pe imaginile de dificultate ridicată')

        plt.axvline(x=163, color='g', label='Schimbarea nivelului de dificultate')
        plt.legend(bbox_to_anchor=(0.205, 1), loc='upper center')

        plt.show()
