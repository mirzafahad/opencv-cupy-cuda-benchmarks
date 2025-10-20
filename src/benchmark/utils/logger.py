import datetime as dt
import logging
import logging.handlers
import os
from logging import CRITICAL, DEBUG, ERROR, INFO, WARNING, Formatter
from typing import Any, Optional

from rich.console import Console


class Handler(logging.Handler):
    """
    Handler that colorizes output but doesn't soft-wrap the output
    """

    # Define information for each log here: https://docs.python.org/3/library/logging.html#logrecord-attributes
    fmt = "| {module:.20}:{funcName:.20}:L{lineno:04d} | {message}"

    # Styles for each type of log, defined by Rich: https://rich.readthedocs.io/en/stable/style.html
    c = {
        DEBUG: "white",
        INFO: "green",
        WARNING: "yellow",
        ERROR: "red",
        CRITICAL: "bold black on red",
    }

    # Save a few milliseconds by initializing the formatters all at once
    # Colors apply to the whole string.
    formatters = {
        DEBUG: Formatter(fmt=f"[{c[DEBUG]}]{fmt}", style="{"),
        INFO: Formatter(fmt=f"[{c[INFO]}]{fmt}", style="{"),
        WARNING: Formatter(fmt=f"[{c[WARNING]}]{fmt}", style="{"),
        ERROR: Formatter(fmt=f"[{c[ERROR]}]{fmt}", style="{"),
        CRITICAL: Formatter(fmt=f"[{c[CRITICAL]}]{fmt}", style="{"),
    }

    def __init__(self):
        super().__init__()
        # Define a console for the logger that is used to pretty-print stuff. Soft-wrap to prevent breaks.
        self.console = Console(
            color_system="256",
            soft_wrap=True,
        )

    def emit(self, record):
        # Emit the record (log) to the console; replace newline with a unicode character.
        created_time = dt.datetime.fromtimestamp(record.created)
        timestamp = f"[{created_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] "
        log_line = timestamp + self.format(record)
        self.console.print(log_line)

    def format(self, record):
        # Apply the default formatting from logging module
        formatter = Handler.formatters.get(record.levelno)
        formatted = formatter.format(record)
        return formatted


class Logger:
    # Holds the logger instance.
    _logger = None

    def __new__(cls, logger_name: str = "default"):
        if cls._logger is None:
            # Initialize the logger.
            logger = logging.getLogger(logger_name)
            logger.addHandler(Handler())
            logger.setLevel(logging.INFO)

            # Set the logger level based on environment variable
            if int(os.getenv("ENABLE_DEBUG", default="0")) == 1:
                logger.setLevel(logging.DEBUG)

            cls._logger = logger

        return cls._logger


    @classmethod
    def set_logger_level(cls, level: int) -> None:
        """
        Set the logging level dynamically

        Args:
            level: integer logging level (ie: DEBUG = 10, CRITICAL = 50)

        """
        cls._logger.setLevel(level)

    def __getattr__(self, name):
        """
        This is added so that pylint doesn't complain about
        "can't find attributes" because pylint doesn't know we
        are returning logging instance.
        """
        pass


if __name__ == "__main__":
    # Create a logger
    os.environ["ENABLE_DEBUG"] = "1"
    logger = Logger()

    # Different types of data
    short_string = "Buzz Buzz"
    long_string = (
        "According to all known laws of aviation there is no way a bee should be able to fly. Its wings are "
        "too small to get its fat little body off the ground. The bee, of course, flies anyway because "
        "bees don't care what humans think is impossible."
    )
    json_obj = {"Bee": "Movie", "BUZZ": "BUZZ\n BUZZ"}
    number = 123456
    sample_list = [1, 2, 3, "buckle my shoe"]

    # Even supports emojis! See the CLDR names here: https://unicode.org/emoji/charts/full-emoji-list.html
    emojis = ":hot_pepper: \u2705 :cross_mark:"

    logger.debug(f"|{short_string:^20}|")
    logger.info(long_string)
    logger.warning(json_obj)
    logger.warning(emojis)
    logger.error(number)
    logger.critical(sample_list)

