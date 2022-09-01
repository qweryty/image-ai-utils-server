import logging

from settings import settings

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'default': {
            'level': settings.LOG_LEVEL.upper(),
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',
            'formatter': 'verbose'
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
        'fastapi': {'level': 'INFO', 'handlers': ['default']},
        'uvicorn.error': {'level': 'INFO', 'handlers': ['default'], 'propagate': False},
        'uvicorn.access': {'level': 'INFO', 'handlers': ['default'], 'propagate': False},
        'uvicorn': {'level': 'INFO', 'handlers': ['default'], 'propagate': False},
        '': {
            'level': settings.LOG_LEVEL.upper(),
            'handlers': ['default'],
            'propagate': True,
        },

    }
}

logging.config.dictConfig(LOGGING)
