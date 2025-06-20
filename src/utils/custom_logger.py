import logging


# define a function that will return a logger object

def get_logger():
    # Set up logging configuration
    logging.basicConfig(
        filename='logs/log.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Create a logger instance
    logger = logging.getLogger(__name__)
    return logger
