import logging
import os


# ------------------------
# Create Logs Folder
# ------------------------
logs_path = "logs"

os.makedirs(
    logs_path,
    exist_ok=True
)


# ------------------------
# Log File Path
# ------------------------
log_file = os.path.join(
    logs_path,
    "api.log"
)


# ------------------------
# Configure Logging
# ------------------------
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="[ %(asctime)s ] "
    "%(levelname)s "
    "%(message)s"
)


# ------------------------
# Create Logger
# ------------------------
logger = logging.getLogger(
    "api_logger"
)
