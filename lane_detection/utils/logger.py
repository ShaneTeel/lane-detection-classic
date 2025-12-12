import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logging(log_level=logging.INFO,
                  log_to_file=True,
                  log_dir="../logs",
                  console_output=True):
    '''
    Description
    -----------
    Configure logging for application

    Parameters
    ----------
    log_level : `logging.DEBUG` / `INFO` / `WARNING` / `ERROR` / `CRITICAL`
        Minimum level to log
    log_to_file : bool, default = True
        Whether to write logs to file
    log_dir : str, default = "logs"
        Location to log files
    console_output : bool
        Whether to display log outputs to console
    
    Returns
    -------
    Root Logger instance
    '''    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers (if any)
    root_logger.handlers.clear()

    # Define format
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_path / f"lane_detection_{timestamp}.log"

        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        root_logger.info(f"Logging to file: {log_file}")
    
    return root_logger

def get_logger(name):
    '''
    Description
    -----------

    Paramters
    ---------
    name
        Module name passed as __name__
    
    Returns
    -------
        Logger Instance
    '''
    return logging.getLogger(name)