"""
This script configures a basic logging setup for an application.

- A logger is created using the module's `__name__`, allowing logs 
  to be traced back to the module where they originate.
- The logger and handler are set to the INFO level, meaning that only 
  messages with a severity level of INFO or higher will be logged.
- A `StreamHandler` is used to direct log messages to the console (standard output).
- A log message formatter is applied to define the structure of log messages, 
  including the timestamp, logger name, log level, and the log message itself.

Usage:
    Use the `logger` instance to log messages in your code:
        logger.info("This is an informational message")
        logger.error("This is an error message")
"""

import logging

# Create a logger
logger = logging.getLogger(__name__)

# Set the log level
logger.setLevel(logging.INFO)

# Create a log message handler
handler = logging.StreamHandler()

# Set the handler level
handler.setLevel(logging.INFO)

# Create a log message formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Set the handler formatter
handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(handler)
