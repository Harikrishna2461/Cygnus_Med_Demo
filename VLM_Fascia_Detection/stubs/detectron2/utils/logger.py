import logging
import time

def setup_logger(output=None, distributed_rank=0, *, color=True, name='detectron2', abbrev_name=None):
    return logging.getLogger(name)

def log_every_n_seconds(lvl, msg, n=1, *, name=None):
    logger = logging.getLogger(name or 'detectron2')
    logger.log(lvl, msg)

def create_small_table(small_dict):
    return str(small_dict)
