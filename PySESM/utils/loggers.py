import logging

def setup_logger() -> logging.Logger:
    # Configure root logger
    logging.basicConfig(level=logging.DEBUG)

    # Create logger
    logger = logging.getLogger('logger')
    logger.setLevel(logging.WARNING)

    # Create file handler
    file_handler = logging.FileHandler('log.log')
    file_handler.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add file handler to logger
    logger.addHandler(file_handler)

    return logger
