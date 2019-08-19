"""
Reference: 
Usage:

About: Class to create central logging class

Author: Satish Jasthi
"""

import logging
import logging.handlers
from pathlib import Path


class DyLogger():
    """
    Class for central logging based
    logger_name: name of loger file
    logger_level: logging.INFO, logging.DEBUG, etc
    """
    def __init__(self, logger_name:str=None, logging_level:int=None):
        if logger_name is None:
            self.LOG_FILE = Path(__file__).resolve().parent / 'logs.txt'
        else:
            self.LOG_FILE = Path(__file__).resolve().parent/f'{logger_name}.txt'
        self.LOGGING_LEVEL = logging_level
        self.LOG_FORMAT = "%(levelname)s %(filename)s %(module)s %(funcName)s %(asctime)s - %(message)s"

        logging.basicConfig(filename=self.LOG_FILE,
                                           level=self.LOGGING_LEVEL,
                                           format = self.LOG_FORMAT,
                                           filemode='w'
                                           )
        # Add the log message handler to the logger to create new log file once file size exceeds 10 mb
        self.handler = logging.handlers.RotatingFileHandler(self.LOG_FILE,
                                                       maxBytes=10**7,
                                                       backupCount=5)

        # logging.addHandler(handler)

    def get_logger(self, name):
        logging.getLogger(name).addHandler(self.handler)
        return logging.getLogger(name)
