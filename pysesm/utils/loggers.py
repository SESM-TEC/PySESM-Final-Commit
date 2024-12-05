import logging


def setup_logger() -> logging.Logger:
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    # Create file handler
    file_handler = logging.FileHandler('log.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Set up the basic configuration
    logging.basicConfig(level=logging.INFO, handlers=[console_handler, file_handler])

    # Create logger
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)

    return logger
