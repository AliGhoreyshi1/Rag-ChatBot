import os
import pandas as pd

DATASET_DOWNLOAD_URL = (
    "https://drive.google.com/file/d/1sgUhjUKzC3pNoQX3-_vgXFgI1l0371Hy/view?usp=sharing"
)

def ensure_dataset_available(csv_path):
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
            url=DATASET_DOWNLOAD_URL,
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
        print("\n--- subgroup ---")
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