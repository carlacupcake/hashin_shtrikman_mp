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
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Set the handler formatter
handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(handler)