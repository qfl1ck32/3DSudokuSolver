from packages.imaging import read_image, show_image

from solution.sudoku_cv import SudokuCV
from solution.sudoku_solver import SudokuSolver

from solution.globals import constants

def main():
    for i in range(1, 326, 1):
        print(f"Solving Cube {i}")

        image = read_image(f"{constants.images_path}/deskewed/{i}.jpg")

        show_image(image, "Input image")

        sudoku_cv = SudokuCV(image, True)

        sudoku_cv.setup()

        sudoku_cv.extract_segments()

        sudoku_cv.extract_paths()

        sudoku_cv.extract_digits_for_faces()

        solver = SudokuSolver(sudoku_cv)

        solved = solver.solve()

        print(f"Solved! Solution found: {solved}")

        solved_image = sudoku_cv.generate_solution_image(solver.faces)

        show_image(solved_image, "Solved image")

if __name__ == '__main__':
    main()
