import requests
import zipfile
import os
from pathlib import Path

def download_montgomery():
    """Download Montgomery County Chest X-ray dataset (small, with masks)."""
    url = "https://public.cancerimagingarchive.net/nbia-search/download?CollectionName=MC-ChestXray"
    print("Montgomery dataset requires manual download from:")
    print("https://www.kaggle.com/datasets/kmader/montgomery-county-chest-xray")
    print("\nOr use the COVID-19 Radiography Database directly from Kaggle:")
    print("1. Go to: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database")
    print("2. Click 'Download' (requires Kaggle account)")
    print("3. Extract the ZIP file to: C:/Users/Pranav/real_data/")

if __name__ == "__main__":
    download_montgomery()
