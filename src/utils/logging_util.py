# Dependencies
import sys
import time
import logging
import functools
import pandas as pd
from pathlib import Path
from typing import Literal
from typing import Optional
from datetime import datetime


# ANSI color codes for console output
class LogColors:
    """
    ANSI color codes for terminal output
    """
    RESET   = "\033[0m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"
    BOLD    = "\033[1m"


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter with color-coded log levels for console output
    """
    LEVEL_COLORS = {logging.DEBUG    : LogColors.CYAN,
                    logging.INFO     : LogColors.GREEN,
                    logging.WARNING  : LogColors.YELLOW,
                    logging.ERROR    : LogColors.RED,
                    logging.CRITICAL : LogColors.RED + LogColors.BOLD,
                   }
    

    def format(self, record):
        # Add color to level name
        levelname = record.levelname

        if (record.levelno in self.LEVEL_COLORS):
            levelname_color  = (f"{self.LEVEL_COLORS[record.levelno]}{levelname}{LogColors.RESET}")
            record.levelname = levelname_color
        
        return super().format(record)


def setup_logger(module_name: Literal['generation', 'eda', 'detection'], log_level: str = 'INFO', log_dir: Path = Path('logs'), 
                 console_output: bool = True, file_output: bool = True) -> logging.Logger:
    """
    Set up a logger with consistent formatting for a specific module
    
    Arguments:
    ----------
        module_name     { Literal } : Name of the module ('generation', 'eda', 'detection')
        
        log_level         { str }   : Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        
        log_dir           { Path }  : Base directory for log files
        
        console_output    { bool }  : Enable console logging
        
        file_output       { bool }  : Enable file logging
    
    Returns:
    --------
          { logging.Logger }        : Configured logger instance
    """
    # Create module-specific log directory
    module_log_dir = log_dir / module_name
    module_log_dir.mkdir(parents  = True, 
                         exist_ok = True,
                        )
    
    # Create logger
    logger = logging.getLogger(f'phq9_analysis.{module_name}')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Define log format
    log_format  = ("%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s")
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Console handler with colors
    if console_output:
        console_handler   = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        
        # Use colored formatter for console
        colored_formatter = ColoredFormatter(log_format, datefmt = date_format)
        console_handler.setFormatter(colored_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if file_output:
        timestamp      = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file       = module_log_dir / f"{module_name}_{timestamp}.log"
        
        file_handler   = logging.FileHandler(log_file, mode = 'w', encoding = 'utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Use standard formatter for file (no colors)
        file_formatter = logging.Formatter(log_format, datefmt=date_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(module_name: str) -> logging.Logger:
    """
    Get existing logger for a module
    
    Arguments:
    ----------
        module_name { str } : Name of the module
    
    Returns:
    --------
        { logging.Logger }  : Logger instance
    """
    return logging.getLogger(f'phq9_analysis.{module_name}')


def log_section_header(logger: logging.Logger, title: str, width: int = 80):
    """
    Log a formatted section header
    
    Arguments:
    ----------
        logger { logging.Logger } : Logger instance

        title       { str }       : Section title
        
        width       { int }       : Width of the header line
    """
    separator = "=" * width
    logger.info(separator)
    logger.info(title.center(width))
    logger.info(separator)


def log_parameters(logger: logging.Logger, params: dict, title: str = "Parameters"):
    """
    Log a dictionary of parameters in a formatted way
    
    Arguments:
    ----------
        logger { logging.Logger } : Logger instance
        
        params     { dict }       : Dictionary of parameters to log
        
        title      { str }        : Title for the parameter section
    """
    logger.info(f"\n{title}:")
    max_key_len = max(len(str(k)) for k in params.keys())
    
    for key, value in params.items():
        logger.info(f"  {str(key).ljust(max_key_len)} : {value}")


def log_dataframe_info(logger: logging.Logger, df: pd.DataFrame, name: str = "DataFrame"):
    """
    Log information about a pandas DataFrame
    
    Arguments:
    ----------
        logger { logging.Logger } : Logger instance
        
        df      { pd.DataFrame }  : pandas DataFrame
        
        name         { str }      : Name/description of the DataFrame
    """
    if not isinstance(df, pd.DataFrame):
        logger.warning(f"{name} is not a pandas DataFrame")
        return
    
    logger.info(f"\n{name} Information:")
    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Columns: {list(df.columns)}")
    logger.info(f"  Missing values: {df.isna().sum().sum()}")
    logger.info(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")


def log_execution_time(logger: logging.Logger):
    """
    Decorator to log execution time of a function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            logger.info(f"Starting {func.__name__}...")
            
            try:
                result       = func(*args, **kwargs)
                elapsed_time = time.time() - start_time

                logger.info(f"Completed {func.__name__} in {elapsed_time:.2f} seconds")
                
                return result
            
            except Exception as e:
                elapsed_time = time.time() - start_time
                logger.error(f"Failed {func.__name__} after {elapsed_time:.2f} seconds: {e}")
                raise
        
        return wrapper
    
    return decorator
