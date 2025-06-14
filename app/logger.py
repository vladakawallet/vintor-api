import logging 
from logging.handlers import RotatingFileHandler
import os

def setup_logger(name, log_file):

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    handler = RotatingFileHandler(log_file, maxBytes=1_000_000, backupCount=5)
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(console)
    logger.propagate = False

    return logger