import logging
import os
from datetime import datetime
from pathlib import Path

LOG_FILE = "app.log"

# consistently find the project root (assuming logger.py is in src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent 
logs_path = os.path.join(PROJECT_ROOT, "logs")

os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
