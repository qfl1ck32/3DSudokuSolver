import sys

def get_current_function_name():
    return f"[{sys._getframe(1).f_code.co_name}]"
