from solution.sudoku_validation import SudokuValidation

def main():
    sudoku_validation = SudokuValidation()

    sudoku_validation.calculate_timing_statistics()
    sudoku_validation.plot_timing()

if __name__ == '__main__':
    main()
