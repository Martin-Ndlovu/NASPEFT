import logging
import sys
from accelerate import Accelerator

def setup_logging(save_path, mode='a'):
    """
    Sets up logging for multiple processes. Only enable logging for the main process.
    """
    accelerator = Accelerator()
    logging.root.handlers = []

    logger = logging.getLogger()
    logger.setLevel(logging.INFO if accelerator.is_main_process else logging.WARNING)
    logger.propagate = False
    print_plain_formatter = logging.Formatter(
        "[%(asctime)s]: %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )
    fh_plain_formatter = logging.Formatter("%(message)s")

    if accelerator.is_main_process:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(print_plain_formatter)
        logger.addHandler(ch)

        if save_path is not None:
            fh = logging.FileHandler(save_path, mode=mode)
            fh.setLevel(logging.INFO)
            fh.setFormatter(fh_plain_formatter)
            logger.addHandler(fh)

def get_logger(name):
    return logging.getLogger(name)