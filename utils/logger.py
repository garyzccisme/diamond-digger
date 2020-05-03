import logging
import sys


# TODO: Add functionality to output logs to a file.
def get_logger(name, level=logging.INFO, context=None):
    """
    Get a custom logger.
    This function should be used to generate loggers within the codebase. The purpose is to unify the
    logging style across the package components for consistent auditing by end-users. This adds
    some customization to the default logger by specifying the caller (file name and line number) to make
    tracking pipeline progress easier during iteration and exposing unexpected behavior with more
    robust debugging information.

    To use:
        from utils import get_logger
        # pass the module name and logging level
        LOGGER = get_logger(__name__, logging.DEBUG)
        # use the logger throughout code as normal
        LOGGER.info("some information about this program")
        LOGGER.warning("a warning about something bad")
        LOGGER.debug("extra stuff for developers")

    Args:
        name: str, the name of the logger
        level: enum, the logging level using the logging level enums, e.g. logging.INFO or logging.DEBUG
        context: str, a contextual string which prepends the logging messages

    Returns: a ``logging`` logger with some added context

    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    fmt = "%(asctime)s %(levelname)-4s %(filename)s:%(lineno)d - %(message)s"
    if context is not None:
        fmt = context + " " + fmt
    formatter = logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S")

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

