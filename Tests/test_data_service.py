import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from Services.data_service import (
    DATASET_COMMENTS_FULL_DOWNLOAD_URL,
    ensure_dataset_available,
    read_csv,
)

URL = DATASET_COMMENTS_FULL_DOWNLOAD_URL


def _fake_gdown(monkeypatch, download_fn):
    monkeypatch.setitem(sys.modules, "gdown", SimpleNamespace(download=download_fn))


# ---------- ensure_dataset_available ----------

def test_existing_dataset_is_returned_without_download(tmp_path, monkeypatch):
    csv_path = tmp_path / "dataset.csv"
    csv_path.write_text("text\nhello", encoding="utf-8")
    mock_download = Mock()
    _fake_gdown(monkeypatch, mock_download)

    assert ensure_dataset_available(str(csv_path), URL) == str(csv_path)
    mock_download.assert_not_called()


def test_successful_download(tmp_path, monkeypatch):
    csv_path = tmp_path / "dataset.csv"

    def fake_download(url, output, quiet):
        with open(output, "w", encoding="utf-8") as f:
            f.write("text\nhello")
        return output

    mock_download = Mock(side_effect=fake_download)
    _fake_gdown(monkeypatch, mock_download)

    assert ensure_dataset_available(str(csv_path), URL) == str(csv_path)
    assert csv_path.stat().st_size > 0
    mock_download.assert_called_once_with(url=URL, output=str(csv_path), quiet=False)


def test_missing_file_without_url_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        ensure_dataset_available(str(tmp_path / "missing.csv"), None)


def test_gdown_not_installed(tmp_path, monkeypatch):
    csv_path = tmp_path / "dataset.csv"
    monkeypatch.setitem(sys.modules, "gdown", None)

    with pytest.raises(RuntimeError, match="gdown is not installed"):
        ensure_dataset_available(str(csv_path), URL)
    assert not csv_path.exists()


def test_download_exception_removes_partial_file(tmp_path, monkeypatch):
    csv_path = tmp_path / "dataset.csv"

    def fake_download(url, output, quiet):
        with open(output, "w", encoding="utf-8") as f:
            f.write("partial data")
        raise ConnectionError("Network error")

    _fake_gdown(monkeypatch, fake_download)

    with pytest.raises(RuntimeError, match="Could not download") as exc_info:
        ensure_dataset_available(str(csv_path), URL)

    assert isinstance(exc_info.value.__cause__, ConnectionError)
    assert not csv_path.exists()


def test_download_returns_none(tmp_path, monkeypatch):
    csv_path = tmp_path / "dataset.csv"
    _fake_gdown(monkeypatch, Mock(return_value=None))

    with pytest.raises(RuntimeError, match="did not return the dataset"):
        ensure_dataset_available(str(csv_path), URL)
    assert not csv_path.exists()


def test_download_produces_empty_file(tmp_path, monkeypatch):
    csv_path = tmp_path / "dataset.csv"

    def fake_download(url, output, quiet):
        open(output, "w").close()
        return output

    _fake_gdown(monkeypatch, fake_download)

    with pytest.raises(RuntimeError, match="did not return the dataset"):
        ensure_dataset_available(str(csv_path), URL)
    assert not csv_path.exists()


def test_creates_missing_parent_directories(tmp_path, monkeypatch):
    csv_path = tmp_path / "data" / "files" / "dataset.csv"

    def fake_download(url, output, quiet):
        with open(output, "w", encoding="utf-8") as f:
            f.write("text\nhello")
        return output

    _fake_gdown(monkeypatch, fake_download)

    assert ensure_dataset_available(str(csv_path), URL) == str(csv_path)
    assert csv_path.parent.exists()


# ---------- read_csv ----------

def test_read_csv_reads_utf8_sig_persian(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("doctor_id,text\n1,عالی بود\n", encoding="utf-8-sig")

    df = read_csv(str(path))

    assert df.loc[0, "text"] == "عالی بود"


def test_read_csv_returns_only_required_columns(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"a": [1], "b": [2], "c": [3]}).to_csv(path, index=False)

    df = read_csv(str(path), ["c", "a"])

    assert list(df.columns) == ["c", "a"]


def test_read_csv_missing_column_raises_key_error(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"username": ["Ali"]}).to_csv(path, index=False)

    with pytest.raises(KeyError, match="Missing columns"):
        read_csv(str(path), ["name"])


def test_read_csv_without_required_returns_everything(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(path, index=False)

    assert list(read_csv(str(path)).columns) == ["a", "b"]