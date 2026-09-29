import os
from pathlib import Path

home_dir = Path(__file__).resolve().parent.parent



import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_YAML_PATH = BASE_DIR / "config" / "config.yaml"


def load_config():
    with open(CONFIG_YAML_PATH, "r") as f:
        return yaml.safe_load(f)

# GET PARENT DIR PATH
DATA_BASE_DIR_PATH = os.path.join(home_dir , "data")
os.makedirs(DATA_BASE_DIR_PATH , exist_ok=True)

ALL_DATA_FILE_PATH = os.path.join(DATA_BASE_DIR_PATH ,"structure_data", "all_data.json")
SAMPLE_DATA_FILE_PATH = os.path.join(DATA_BASE_DIR_PATH ,"structure_data", "sample_data.json")


TRAIN_DATA_FILE_PATH = os.path.join(DATA_BASE_DIR_PATH , "train_data.json")
TEST_DATA_FILE_PATH = os.path.join(DATA_BASE_DIR_PATH , "test_data.json")
VAL_DATA_FILE_PATH = os.path.join(DATA_BASE_DIR_PATH , "val_data.json")
THRESHOLD_FILE_PATH = os.path.join(DATA_BASE_DIR_PATH , "thresholds.json")
MODEL_PATH = os.path.join(DATA_BASE_DIR_PATH,"best_model.pt")