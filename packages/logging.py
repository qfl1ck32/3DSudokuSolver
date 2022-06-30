import logging
import sys

logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s/%(levelname)s: %(message)s")

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(formatter)
logger.addHandler(ch)

class StepLogger:
    def __init__(self, steps: int, log_function=logger.info, verbose=True):
        self.steps = steps
        self.current_step = 1

        self.log_function = log_function

        self.verbose = verbose

    def log(self, message: str):
        if not self.verbose:
            return

        self.log_function(f"[{self.current_step} / {self.steps}] {message}")
        self.current_step += 1
