import os
import sys
import logging
from termcolor import colored
from pytorch_lightning.utilities import rank_zero_only


class ColoredFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self.use_color = kwargs.pop("use_color", True)
        self.COLOR = {
            logging.DEBUG: "blue",
            logging.INFO: "green",
            logging.WARNING: "yellow",
            logging.ERROR: "red",
            logging.CRITICAL: "red",
        }
        super(ColoredFormatter, self).__init__(*args, **kwargs)

    def format(self, record):
        levelno = record.levelno
        if self.use_color and levelno in self.COLOR:
            record.levelname = colored(record.levelname, self.COLOR[levelno], attrs=["underline"])
        if record.name == "lightning":
            record.msg = "\n" + record.msg
        # return logging.Formatter.format(self, record)
        return super(ColoredFormatter, self).format(record)


def setup_logger(output=None, logger_name=None, use_color=True):
    logger = logging.getLogger(logger_name) # getLogger(name=None) defaults to root logger
    logger.setLevel(logging.INFO)
    logger.propagate = True #False # do not prop to root logger

    # Formatter
    fmt = ["[%(asctime)s] %(name)s %(levelname)s: ", "%(message)s"]
    datefmt = "%m/%d %H:%M:%S"
    file_formatter = logging.Formatter(fmt[0]+fmt[1], datefmt=datefmt)
    stream_formatter = (ColoredFormatter(colored(fmt[0], "green") + fmt[1],
                                         datefmt=datefmt, use_color=True)
                        if use_color else file_formatter)

    # Stream handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(stream_formatter)
    logger.addHandler(sh)

    # File handler
    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "log.txt")
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        fh = logging.FileHandler(filename)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(file_formatter)
        logger.addHandler(fh)

    # Rank 1 only? in the main
    logger.info = rank_zero_only(logger.info)

    return logger


if __name__ == "__main__":
    logger1 = setup_logger()
    logger1.setLevel(logging.DEBUG)
    logger1.debug("debug")
    logger1.info("info")
    logger1.warning("warning")
    logger1.error("error")
    logger1.critical("critical")

    logger2 = setup_logger(output="output2", logger_name="logger2", use_color=False)
    logger2.debug("debug")
    logger2.info("info")
    logger2.warning("warning")
    logger2.error("error")
    logger2.critical("critical")
