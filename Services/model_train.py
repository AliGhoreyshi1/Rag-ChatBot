import os
import pandas as pd
import pickle
import sys
import time
from Helpers.data_pipeline import comments_to_documents, build_training_data, split_packed


DATASET_COMMENTS_FULL_DOWNLOAD_URL = (
    "https://drive.google.com/file/d/1sgUhjUKzC3pNoQX3-_vgXFgI1l0371Hy/view?usp=sharing"
)
DATASET_DOCTORS_RAW_CANONICAL_DOWNLOAD_URL = (
    "https://drive.google.com/file/d/1pNWjDf2h9BhFTlD-9_b1FaPNFx3hdj6L/view?usp=sharing"
)
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
DATA_DIR = os.path.join(BASE_DIR, "Data")
COMMENTS_FULL_CSV_PATH = os.path.join(DATA_DIR, "comments_full.csv")
DOCTORS_RAW_CANONICAL_CSV_PATH = os.path.join(DATA_DIR, "doctors_raw_canonical.csv")
NUM_MERGES = int(sys.argv[1]) if len(sys.argv) > 1 else 50
SEQ_LENGTH = 128

def ensure_dataset_available(csv_path, download_path):
    """Return a usable CSV path, downloading the shared dataset if it is missing.

    The download is performed lazily so users who provide their own existing CSV
    do not need the optional gdown package.
    """
    if os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0:
        return csv_path

    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "The dataset file was not found and the 'gdown' package is not installed. "
            "Install it with 'python -m pip install gdown', or download the CSV manually "
            "from the Google Drive link in README.md."
        ) from exc

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    print(f"Dataset not found at '{csv_path}'. Downloading it from Google Drive...")

    try:
        downloaded_path = gdown.download(
            url=download_path,
            output=csv_path,
            quiet=False,
            
        )
    except Exception as exc:
        if os.path.isfile(csv_path):
            os.remove(csv_path)
        raise RuntimeError(
            "Could not download the dataset from Google Drive. Check your internet "
            "connection and make sure the file is shared with 'Anyone with the link'."
        ) from exc

    if not downloaded_path or not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0:
        if os.path.isfile(csv_path):
            os.remove(csv_path)
        raise RuntimeError(
            "Google Drive did not return the dataset. Check the sharing permissions "
            "and download the CSV manually using the instructions in README.md."
        )

    print(f"Dataset downloaded to: {csv_path}")
    return csv_path

def load_dataset(csv_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path)
        print(df.columns.tolist(), df.shape)
        print("\n--- doctors ---")
        print(df['name'].value_counts())

        return df

    except KeyError as e:
        raise KeyError(
            f"Required column not found in dataset: {e}"
        ) from e

    except IndexError as e:
        raise IndexError(
            f"Index is out of range while accessing dataset: {e}"
        ) from e

    except ValueError as e:
        raise ValueError(
            f"Invalid value encountered while loading the CSV dataset: {e}"
        ) from e

    except TypeError as e:
        raise TypeError(
            f"Invalid data type provided for CSV loading: {e}"
        ) from e

    except AttributeError as e:
        raise AttributeError(
            f"An attribute error occurred while processing the dataset: {e}"
        ) from e

def bpe_on_data(csv_path: str):
    start = time.time()
    docs = comments_to_documents(csv_path, text_column="text", group_column="doctor_id", min_words=60)
    print(f"Documents from comments: {len(docs)}", flush=True)

    tokenizer, packed, final_docs = build_training_data(docs, num_merges=NUM_MERGES, seq_length=SEQ_LENGTH)
    print(f"Final documents: {len(final_docs)}", flush=True)
    print(f"Vocab size: {tokenizer.vocab_size()}", flush=True)
    print(f"Training blocks: {len(packed['input_ids'])}", flush=True)

    train, val = split_packed(packed, val_fraction=0.15)
    print(f"Train blocks: {len(train['input_ids'])}, Val blocks: {len(val['input_ids'])}", flush=True)

    sample = final_docs[0][:200]
    ids = tokenizer.encode(sample)
    print("Roundtrip OK:", tokenizer.decode(ids) == sample, flush=True)
    print(f"Sample tokens: {len(ids)} for {len(sample.encode('utf-8'))} bytes", flush=True)

    out = f"tokenizer_{NUM_MERGES}.pkl"
    with open(out, "wb") as f:
        pickle.dump(tokenizer, f)
    with open(f"packed_{NUM_MERGES}.pkl", "wb") as f:
        pickle.dump({"train": train, "val": val}, f)
    print(f"Saved {out}. Done in {time.time() - start:.0f}s", flush=True)

def main():
    csv_abs_path = ensure_dataset_available(COMMENTS_FULL_CSV_PATH, DATASET_COMMENTS_FULL_DOWNLOAD_URL)
    csv_doctors_raw = ensure_dataset_available(DOCTORS_RAW_CANONICAL_CSV_PATH, DATASET_DOCTORS_RAW_CANONICAL_DOWNLOAD_URL)
    bpe_on_data(csv_abs_path)

if __name__ == "__main__":
    main()