# download_data.py
import os
import zipfile
import urllib.request
from pathlib import Path

def download_and_extract():
    # Defining paths
    data_dir = Path("./data")
    zip_path = data_dir / "PennFudanPed.zip"
    extract_path = data_dir / "PennFudanPed"

    # Checking if data already exists to avoid redundant downloads
    if extract_path.exists():
        print(f"Dataset already exists at: {extract_path.resolve()}")
        return

    # Creating data directory if it doesn't exist
    data_dir.mkdir(parents=True, exist_ok=True)

    # Penn-Fudan Dataset URL
    url = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"
    
    print("Downloading Penn-Fudan Pedestrian Dataset (approx. 50MB)...")
    urllib.request.urlretrieve(url, zip_path)
    print("Download complete!")

    print("Extracting files...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)
    
    # Cleaning up the zip file to keep the repository tidy
    zip_path.unlink()
    print(f"Dataset successfully extracted to: {extract_path.resolve()}")

if __name__ == "__main__":
    download_and_extract()