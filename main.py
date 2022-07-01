from timeit import default_timer
from packages.imaging import read_image, show_image
from packages.logging import logger

from solution.sudoku_cv import SudokuCV
from solution.sudoku_solver import SudokuSolver

from solution.globals import constants


def main():
    for i in range(1, 326, 1):
        print(f"Solving Cube {i}")

        image = read_image(f"{constants.images_path}/deskewed/{i}.jpg")

        show_image(image, "Input image")

        start_time_info_extraction = default_timer()

        sudoku_cv = SudokuCV(image, True)

        sudoku_cv.setup()

        sudoku_cv.extract_segments()

        sudoku_cv.extract_paths()

        sudoku_cv.extract_digits_for_faces()

        end_time_info_extraction = default_timer()

        solver = SudokuSolver(sudoku_cv)

        start_time_solving = default_timer()

        solved = solver.solve()

        end_time_solving = default_timer()

        info_extraction_time = end_time_info_extraction - start_time_info_extraction
        solving_time = end_time_solving - start_time_solving

        logger.info(f"Solved! Solution found: {solved}")
        logger.info(f"Time for visual info extraction: %.3fs, generating the solution: %.3fs, total: %.3fs" % (info_extraction_time, solving_time, info_extraction_time + solving_time))

        solved_image = sudoku_cv.generate_solution_image(solver.faces)

        show_image(solved_image, "Solved image")

if __name__ == '__main__':
    main()
