"""
Settings for logging.

Variables:
    LOGGING (dict) - logging settings
"""

import logging.config
import os

from settings import SETTINGS

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "default": {
            "level": SETTINGS.LOG_LEVEL.upper(),
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": "verbose",
        },
        "file_handler": {
            "level": SETTINGS.FILE_LOG_LEVEL.upper(),
            "formatter": "verbose",
            "class": "logging.FileHandler",
            "filename": SETTINGS.LOG_FILE_PATH,
        },
        "blackhole": {"level": "DEBUG", "class": "logging.NullHandler"},
    },
    "formatters": {
        "verbose": {
            "format": "%(log_color)s%(asctime)s [%(levelname)s] [%(name)s] %(message)s (%(filename)s:%(lineno)d)",
            "()": "colorlog.ColoredFormatter",
            "log_colors": {
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        }
    },
    "loggers": {
        "fastapi": {"level": "INFO", "handlers": ["default", "file_handler"]},
        "uvicorn.error": {
            "level": "INFO",
            "handlers": ["default", "file_handler"],
            "propagate": False,
        },
        "uvicorn.access": {
            "level": "INFO",
            "handlers": ["default", "file_handler"],
            "propagate": False,
        },
        "uvicorn": {
            "level": "INFO",
            "handlers": ["default", "file_handler"],
            "propagate": False,
        },
        "": {
            "level": SETTINGS.LOG_LEVEL.upper(),
            "handlers": ["default", "file_handler"],
            "propagate": True,
        },
    },
}

os.makedirs(os.path.dirname(SETTINGS.LOG_FILE_PATH), exist_ok=True)

logging.config.dictConfig(LOGGING)
