import torch
import functools

def retry_if_cuda_oom(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return func(*args, **kwargs)
    return wrapper
