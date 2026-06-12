# data/download_kaggle.py
"""Utility to download and extract the Kaggle LLM Classification dataset.

Ensures the kaggle library is installed, checks for credentials,
and downloads the dataset to the local data/llm_classification folder.
"""

import os
import sys
import subprocess
import zipfile


def ensure_kaggle_installed():
    """Ensure the kaggle Python library is installed."""
    try:
        import kaggle
    except ImportError:
        print("Kaggle package not found. Installing via pip...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
            print("Successfully installed kaggle package.")
        except Exception as e:
            print(f"Error installing kaggle package: {e}")
            sys.exit(1)


def check_credentials():
    """Check if Kaggle API credentials are set up."""
    # Check environment variables
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True

    # Check ~/.kaggle/kaggle.json
    home = os.path.expanduser("~")
    kaggle_json = os.path.join(home, ".kaggle", "kaggle.json")
    if os.path.exists(kaggle_json):
        return True

    return False


def main():
    print("\n" + "=" * 60)
    print("  Kaggle LLM Classification Dataset Downloader")
    print("=" * 60 + "\n")

    ensure_kaggle_installed()

    if not check_credentials():
        print("ERROR: Kaggle credentials not found!")
        print("Please follow one of these setup steps:")
        print("1. Download 'kaggle.json' from your Kaggle profile page (Account tab) and place it in:")
        home = os.path.expanduser("~")
        print(f"   {os.path.join(home, '.kaggle', 'kaggle.json')}")
        print("2. Set the following environment variables in your terminal/system:")
        print("   $env:KAGGLE_USERNAME=\"your_username\"")
        print("   $env:KAGGLE_KEY=\"your_api_key\"")
        print("\nOnce set up, run this script again.")
        sys.exit(1)

    # Download directory
    output_dir = os.path.join("data", "llm_classification")
    os.makedirs(output_dir, exist_ok=True)

    print("Authenticating with Kaggle API...")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        print(f"Authentication failed: {e}")
        print("Please verify that your kaggle.json credentials are correct and not expired.")
        sys.exit(1)

    competition_name = "llm-classification-finetuning"
    zip_filename = f"{competition_name}.zip"
    zip_path = os.path.join(output_dir, zip_filename)

    # Skip download if files already exist
    train_csv = os.path.join(output_dir, "train.csv")
    test_csv = os.path.join(output_dir, "test.csv")
    if os.path.exists(train_csv) and os.path.exists(test_csv):
        print(f"Dataset already exists and is extracted at: {output_dir}")
        return

    print(f"Downloading files for competition '{competition_name}'...")
    try:
        api.competition_download_files(competition_name, path=output_dir)
        print("Download complete.")
    except Exception as e:
        print(f"Download failed: {e}")
        print("Verify you have accepted the competition rules on Kaggle's website.")
        sys.exit(1)

    if os.path.exists(zip_path):
        print(f"Extracting {zip_filename} to {output_dir}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            os.remove(zip_path)
            print("Extraction complete. Cleaned up zip file.")
        except Exception as e:
            print(f"Extraction failed: {e}")
            sys.exit(1)
    else:
        # Check if files downloaded directly
        if os.path.exists(train_csv):
            print("Files downloaded successfully (already extracted).")
        else:
            print(f"Warning: Expected zip file at {zip_path} not found.")

    print("\nDataset files available:")
    for f in os.listdir(output_dir):
        fpath = os.path.join(output_dir, f)
        print(f"  - {f} ({os.path.getsize(fpath) / (1024*1024):.2f} MB)")
    print()


if __name__ == "__main__":
    main()
