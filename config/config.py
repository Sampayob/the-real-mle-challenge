from pathlib import Path


# Directories
DIR_REPO = Path(__file__).parent.parent.absolute()
DIR_CONFIG = Path(DIR_REPO, "config")
DIR_MODELS = Path(DIR_REPO, "models")
DIR_DATA = Path(DIR_REPO) / "data"
DIR_DATA_RAW = Path(DIR_REPO) / "data" / "raw"
DIR_DATA_PROCESSED = Path(DIR_REPO) / "data" / "processed"

# Filepaths
FILEPATH_DATA_RAW = DIR_DATA_RAW / "listings.csv"
FILEPATH_DATA_PROCESSED = DIR_DATA_PROCESSED / "preprocessed_listings.csv"
FILEPATH_ARGS = DIR_CONFIG / "args.json"
FILEPATH_MODEL = DIR_MODELS / "simple_classifier.pkl"

# Create dirs
DIR_CONFIG.mkdir(parents=True, exist_ok=True)
DIR_MODELS.mkdir(parents=True, exist_ok=True)
DIR_DATA_RAW.mkdir(parents=True, exist_ok=True)
DIR_DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
