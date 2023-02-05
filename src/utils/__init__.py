import logging
import os

from src.utils import tempel_logger

if os.environ.get('LOGGING_LEVEL') is not None and os.environ.get('LOGGING_LEVEL').strip() != '':
    tempel_logger.logger_level = logging._nameToLevel[os.environ.get('LOGGING_LEVEL')]
else:
    tempel_logger.logger_level = logging.INFO
    # tempel_logger.logger_level = logging.DEBUG

print('logger_level assigned: %s' % tempel_logger.logger_level)
