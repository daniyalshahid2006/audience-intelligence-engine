import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from text_cleaning import clean_comments
from content_analysis import analyse_content
from comment_clustering import cluster_comments
from anomaly_detection import detect_anomalies
from recommendations import build_recommendations

if __name__ == "__main__":
    print("=== Step 1: text cleaning ===")
    clean_comments()
    print("\n=== Step 2: Q1 content analysis ===")
    analyse_content()
    print("\n=== Step 3: Q2 and Q4 comment clustering ===")
    cluster_comments()
    print("\n=== Step 4: Q3 anomaly detection ===")
    detect_anomalies()
    print("\n=== Step 5: Q5 recommendations ===")
    build_recommendations()
    print("\nPIPELINE COMPLETE")