import logging
from logging.handlers import RotatingFileHandler
from typing import Literal


Level = Literal['debug', 'warning', 'info', 'error', 'crit']


def log_decorator(func):
    def wrapper(*args, **kwarks):
        func(*args, **kwarks)
        create_log(f'Function {func.__name__} called!')
    return wrapper


def create_log(
    log: Exception|str,
    levelname: Level = 'debug',
    log_file: str = 'logger.log',
    max_bytes: int = 1_000_000,
):
    logger = logging.Logger(__name__)
    handler = RotatingFileHandler(
        log_file, 'a', maxBytes=max_bytes, backupCount=1
    )
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    log_exc = False
    if type(log) != str:
        log_exc = True
    match levelname:
        case 'debug':
            logger.debug(log, exc_info=log_exc)
        case 'warning':
            logger.warning(log, exc_info=log_exc)
        case 'info':
            logger.info(log, exc_info=log_exc)
        case 'error':
            logger.error(log, exc_info=log_exc)
        case 'crit':
            logger.critical(log, exc_info=log_exc)

