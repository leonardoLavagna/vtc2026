import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data.csv")
LOG_DIR = os.path.join(BASE_DIR, "results", "logs")
PLOT_DIR = os.path.join(BASE_DIR, "results", "plots")
FILES_DIR = os.path.join(BASE_DIR, "results", "files")
# create if needed
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(FILES_DIR, exist_ok=True)
