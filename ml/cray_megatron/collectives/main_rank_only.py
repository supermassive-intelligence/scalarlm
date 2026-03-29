from functools import wraps
from gpu_aware_mpi import get_rank, barrier

def is_main_rank():
    return get_rank() == 0

_in_main_rank_only = False

def main_rank_only(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        global _in_main_rank_only

        if _in_main_rank_only:
            return func(*args, **kwargs)

        _in_main_rank_only = True
        try:
            barrier()
            result = func(*args, **kwargs) if is_main_rank() else None
            barrier()
            return result
        finally:
            _in_main_rank_only = False

    return wrapper
