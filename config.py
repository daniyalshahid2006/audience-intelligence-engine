from pathlib import Path

# Project folders
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"

# Reproducibility
RANDOM_SEED = 42

# Simulated data
N_POSTS = 3000
DATA_SOURCE = "simulated"   # never mix with real data without changing this label
COLLECTION_DATE = "2026-11-08"
ANOMALY_RATE = 0.03

# Business setting
PRODUCT_CATEGORY = "Lawn suit"

PLATFORMS = ["Instagram", "TikTok", "Facebook"]
PLATFORM_WEIGHTS = [0.45, 0.25, 0.30]

CONTENT_TYPES = ["Reel", "Product photo", "Influencer post",
                 "Styling video", "Sale announcement", "Eid campaign"]
CONTENT_WEIGHTS = [0.25, 0.25, 0.15, 0.12, 0.13, 0.10]

CAMPAIGNS = ["Eid Collection", "11.11 Collection", "12.12 Collection",
             "Summer Lawn Launch", "Regular Collection"]
CAMPAIGN_WEIGHTS = [0.22, 0.22, 0.18, 0.20, 0.18]

# Fictional brands and a relative audience-size multiplier
BRANDS = {"Brand A": 1.6, "Brand B": 1.2, "Brand C": 1.0,
          "Brand D": 0.7, "Brand E": 0.5}