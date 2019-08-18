"""
Reference: 
Usage:

About: Class to create central logging class

Author: Satish Jasthi
"""

import logging
import logging.handlers
from pathlib import Path


class DyLogger(object):
    """
    Class for central logging based
    logger_name: name of loger file
    logger_level: logging.INFO, logging.DEBUG, etc
    """
    def __init__(self, logger_name:str, logging_level:int):
        self.LOG_FILE = Path(__file__).resolve().parent/f'{logger_name}.txt'
        self.LOGGING_LEVEL = logging_level
        self.LOG_FORMAT = "%(levelname)s %(filename)s %(module)s %(funcName)s %(asctime)s - %(message)s"

        self.logging = logging.basicConfig(filename=self.LOG_FILE,
                                           level=self.LOGGING_LEVEL,
                                           format = self.LOG_FORMAT,
                                           filemode='w'
                                           )
        # Add the log message handler to the logger to create new log file once file size exceeds 10 mb
        handler = logging.handlers.RotatingFileHandler(self.LOG_FILE,
                                                       maxBytes=10**7,
                                                       backupCount=5)

        self.logging.addHandler(handler)

