import logging
import sys
import os

LOG_FILE = "log.log"

# Define a function to set up and return a logger
def get_logger(path, name="main"):
    logger = logging.getLogger(name)

    # Prevent adding multiple handlers in case of multiple imports
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)

        # Explicitly set encoding='utf-8'
        file_handler = logging.FileHandler(os.path.join(path, LOG_FILE), encoding='utf-8')

        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%m/%d/%Y %I:%M:%S %p"
        ))

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))

        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        # Redirect print statements to logger
        sys.stdout = StreamToLogger(logger, logging.INFO)
        sys.stderr = StreamToLogger(logger, logging.ERROR)  # Redirect stderr as well

    return logger

def log_print(logger, *args, **kwargs):
    """
    Prints and logs the message using the provided logger.
    """
    message = " ".join(str(arg) for arg in args)  # Convert args to string
    logger.info(message)  # Log the message

# Helper class to redirect stdout to the logger
class StreamToLogger:
    def __init__(self, logger, log_level):
        self.logger = logger
        self.log_level = log_level
        self.line_buffer = ""

    def write(self, message):
        if message.strip():  # Avoid logging empty lines
            self.logger.log(self.log_level, message.strip())

    def flush(self):
        pass  # No need to flush manually; logging handles it

    def isatty(self):
        return False