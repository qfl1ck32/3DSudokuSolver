from glob import glob
import os
from timeit import default_timer

import cv2 as cv

from solution.globals import constants

from packages.imaging import read_image

from solution.sudoku_cv import SudokuCV

from solution.sudoku_solver import SudokuSolver
from solution.globals import constants


def main_test():
    solve_errors = []
    numbers_errors = []
    paths_errors = []
    extract_edges_errors = []


    for i in range(1, 326, 1):
        if len(glob(f"{constants.solution_path}/images/{i}-*")):
            continue

        if not os.path.exists(f"{constants.images_path}/deskewed/{i}.jpg"):
            continue

        print(f"Solving for Cube {i}")

        start = default_timer()

        image = read_image(f"{constants.images_path}/deskewed/{i}.jpg")

        sudoku_cv = SudokuCV(image, True)

        try:
            sudoku_cv.setup()
        except Exception as e:
            print(e)
            extract_edges_errors.append(i)
            continue

        try:
            sudoku_cv.extract_segments()
        except Exception as e:
            print(e)
            extract_edges_errors.append(i)
            continue

        path_path = f"{constants.solution_path}/paths/paths-{i}.pkl"

        try:
            if os.path.exists(path_path):
                sudoku_cv.load_paths(path_path)
            else:
                sudoku_cv.extract_paths()
                sudoku_cv.save_paths(path_path)

        except Exception as e:
            print(e)

            paths_errors.append(i)
            continue

        try:
            sudoku_cv.extract_digits_for_faces()
        except Exception as e:
            numbers_errors.append(i)
            print(e)
            continue
       
        solver = SudokuSolver(sudoku_cv)

        print("Generating the solution...")

        solved = solver.solve()

        print(f"Solved: {solved}")

        if not solved:
            solve_errors.append(i)

        end = default_timer()

        print(end - start)

        print(f"Done. Recursive calls: {solver.recursive_calls}")

        if not solved:
            continue

        img = sudoku_cv.generate_solution_image(solver.faces)

        cv.imwrite(
            f"{constants.solution_path}/images/{i}-{end - start}.jpg", img)

        print(solve_errors)
        print(numbers_errors)
        print(paths_errors)
        print(extract_edges_errors)

    print(solve_errors)
    print(numbers_errors)
    print(paths_errors)
    print(extract_edges_errors)
