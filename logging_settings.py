import logging.config
import os.path

from settings import settings
from utils import resolve_path

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'default': {
            'level': settings.LOG_LEVEL.upper(),
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',
            'formatter': 'verbose',
        },
        'file_handler': {
            'level': settings.FILE_LOG_LEVEL.upper(),
            'formatter': 'verbose',
            'class': 'logging.FileHandler',
            'filename': resolve_path('messages.log')
        },
        'blackhole': {'level': 'DEBUG', 'class': 'logging.NullHandler'},
    },
    'formatters': {
        'verbose': {
            'format': '%(log_color)s%(asctime)s [%(levelname)s] [%(name)s] %(message)s (%(filename)s:%(lineno)d)',
            '()': 'colorlog.ColoredFormatter',
            'log_colors': {
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            },
        }
    },
    'loggers': {
        'fastapi': {'level': 'INFO', 'handlers': ['default', 'file_handler']},
        'uvicorn.error': {
            'level': 'INFO', 'handlers': ['default', 'file_handler'], 'propagate': False
        },
        'uvicorn.access': {
            'level': 'INFO', 'handlers': ['default', 'file_handler'], 'propagate': False
        },
        'uvicorn': {
            'level': 'INFO', 'handlers': ['default', 'file_handler'], 'propagate': False
        },
        '': {
            'level': settings.LOG_LEVEL.upper(),
            'handlers': ['default', 'file_handler'],
            'propagate': True,
        },

    }
}

logging.config.dictConfig(LOGGING)
