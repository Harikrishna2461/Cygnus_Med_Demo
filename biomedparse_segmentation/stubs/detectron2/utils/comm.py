import torch

def get_world_size(): return 1
def get_rank(): return 0
def is_main_process(): return True
def synchronize(): pass

def all_gather(data, group=None):
    return [data]

def gather(data, dst=0, group=None):
    return [data] if get_rank() == dst else []

def shared_random_seed():
    import random
    return random.randint(0, 2**31 - 1)
