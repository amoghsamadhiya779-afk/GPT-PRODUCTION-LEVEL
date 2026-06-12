# data/download_cosmopedia.py
"""Utility to stream and consolidate the Hugging Face Cosmopedia dataset.

Downloads mathematical and educational textbook subsets (auto_math_text,
khanacademy, openstax) and merges them into a single text corpus file.
"""

import os
import sys
import subprocess


def ensure_dependencies():
    """Ensure Hugging Face datasets and pyarrow libraries are installed."""
    try:
        import datasets
        import pyarrow
    except ImportError:
        print("Hugging Face datasets/pyarrow not found. Installing via pip...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets", "pyarrow"])
            print("Successfully installed datasets and pyarrow.")
        except Exception as e:
            print(f"Error installing dependencies: {e}")
            sys.exit(1)


def main():
    print("\n" + "=" * 60)
    print("  Hugging Face Cosmopedia Math Dataset Downloader")
    print("=" * 60 + "\n")

    ensure_dependencies()

    from datasets import load_dataset

    subsets = ["auto_math_text", "khanacademy", "openstax"]
    output_path = os.path.join("data", "cosmopedia_math.txt")
    os.makedirs("data", exist_ok=True)

    print(f"Consolidating math textbooks to: {output_path}")
    print("Streaming samples from Hugging Face...")

    samples_per_subset = 2000
    merged_count = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        for subset in subsets:
            print(f"  Streaming '{subset}' configuration...")
            try:
                # Load configuration as streaming to avoid downloading massive parquet files completely
                ds = load_dataset("HuggingFaceTB/cosmopedia", subset, split="train", streaming=True)
                count = 0
                for row in ds:
                    text = row.get("text", "").strip()
                    if text:
                        # Append the BPE end-of-text separator for the causal tokenizer
                        out_f.write(text + "\n<|endoftext|>\n")
                        count += 1
                        merged_count += 1

                    if count >= samples_per_subset:
                        break
                print(f"    [OK] Loaded {count} samples from '{subset}'")
            except Exception as e:
                print(f"    [ERROR] Failed to load subset '{subset}': {e}")

    print("\n" + "=" * 60)
    if merged_count > 0:
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  Success: Math corpus generated!")
        print(f"  Total samples merged : {merged_count:,}")
        print(f"  Saved location       : {output_path}")
        print(f"  Corpus file size     : {file_size_mb:.2f} MB")
    else:
        print("  Error: No samples were loaded. Verify internet connection.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
