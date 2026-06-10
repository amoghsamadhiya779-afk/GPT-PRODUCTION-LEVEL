# data/download.py
"""Training data download utility.

Downloads the demo training text ("The Verdict" by Edith Wharton)
from the LLMs-from-scratch repository for pre-training demos.

Usage:
    py data/download.py
    py data/download.py --output data/the-verdict.txt
"""

import argparse
import os
import sys

import requests
from tqdm import tqdm


DEFAULT_URL = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/"
    "main/ch02/01_main-chapter-code/the-verdict.txt"
)
DEFAULT_OUTPUT = os.path.join("data", "the-verdict.txt")


def download_file(url: str, destination: str) -> None:
    """Download a file from a URL with a progress bar.

    Skips the download if the file already exists with the correct size.
    """
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    file_size = int(response.headers.get("Content-Length", 0))

    # Skip if already downloaded
    if os.path.exists(destination):
        local_size = os.path.getsize(destination)
        if file_size and file_size == local_size:
            print(f"  [SKIP] File already exists: {destination}")
            return

    os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)

    desc = os.path.basename(destination)
    with tqdm(total=file_size, unit="iB", unit_scale=True, desc=desc) as pbar:
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    print(f"  [DONE] Saved to {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download training data for GPT pre-training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="URL of the training text file.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Local path to save the downloaded file.",
    )
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("  GPT Training Data Downloader")
    print("=" * 50 + "\n")

    download_file(args.url, args.output)

    # Show file info
    with open(args.output, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"\n  File size   : {len(text):,} characters")
    print(f"  Preview    : {text[:80]}...")
    print()


if __name__ == "__main__":
    main()
