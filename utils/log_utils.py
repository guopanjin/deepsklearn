import logging
from logging.handlers import TimedRotatingFileHandler
import os

class Logger:
    _logger = None
    _projectName="DeepSklearn"
    _logger_level=logging.INFO
    @staticmethod
    def getRootPath(projectName):
        curPath = os.path.abspath(os.path.dirname(__file__))
        rootPath = curPath[:curPath.find(f'{projectName}') + len(f'{projectName}')]
        return rootPath
    @staticmethod
    def get_logger(log_file="app.log",projectName=_projectName, backup_count=30, log_level=_logger_level):
        log_dir=Logger.getRootPath(projectName)
        log_dir = os.path.join(log_dir, "logs")
        if Logger._logger is None:
            Logger._logger = Logger._create_logger(log_file, log_dir, backup_count, log_level)
        return Logger._logger

    @staticmethod
    def _create_logger(log_file, log_dir, backup_count, log_level):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_path = os.path.join(log_dir, log_file)

        logger = logging.getLogger("AppLogger")
        logger.setLevel(log_level)

        handler = TimedRotatingFileHandler(log_path, when="midnight", interval=1, backupCount=backup_count)
        handler.suffix = "%Y-%m-%d"

        formatter = logging.Formatter(
            '%(asctime)s-%(levelname)s-%(filename)s_%(funcName)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)

        logger.addHandler(handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        return logger
if __name__ == '__main__':
    ROOT = os.environ.get("PROJECT_ROOT")
    print(ROOT)
    pass