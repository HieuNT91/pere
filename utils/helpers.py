import logging
import numpy as np

import os
import pickle


def make_logger(name, log_dir):
    log_dir = log_dir or 'logs'
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, 'debug.log')
    handler = logging.FileHandler(log_file)
    formater = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formater)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formater)
    logger.addHandler(stream_handler)

    return logger


def pdump(x, name, outdir='.'):
    with open(os.path.join(outdir, name), mode='wb') as f:
        pickle.dump(x, f)


def pload(name, outdir='.'):
    with open(os.path.join(outdir, name), mode='rb') as f:
        return pickle.load(f)
