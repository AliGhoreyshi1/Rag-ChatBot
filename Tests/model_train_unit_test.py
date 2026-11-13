import os
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from Services.model_train import ensure_dataset_available, DATASET_DOWNLOAD_URL

def test_existing_dataset(tmp_path):
    csv_path = tmp_path / "dataset.csv"
    csv_path.write_text("text\nhello", encoding="utf-8")

    result = ensure_dataset_available(str(csv_path))

    assert result == str(csv_path)
    assert csv_path.exists()

def test_successful_download(tmp_path, monkeypatch):
    csv_path = tmp_path / "dataset.csv"

    def fake_download(url, output, quiet):
        with open(output, "w", encoding="utf-8") as file:
            file.write("text\nhello")

        return output

    mock_download = Mock(side_effect=fake_download)

    monkeypatch.setitem(
        sys.modules,
        "gdown",
        SimpleNamespace(download=mock_download)
    )

    result = ensure_dataset_available(str(csv_path))

    assert result == str(csv_path)
    assert csv_path.exists()
    assert csv_path.stat().st_size > 0

    mock_download.assert_called_once_with(
        url=DATASET_DOWNLOAD_URL,
        output=str(csv_path),
        quiet=False
    )

def test_gdown_not_installed(tmp_path, monkeypatch):
    csv_path = tmp_path / "dataset.csv"

    monkeypatch.setitem(sys.modules, "gdown", None)

    with pytest.raises(RuntimeError, match="gdown.*not installed"):
        ensure_dataset_available(str(csv_path))

    assert not csv_path.exists()

def test_download_exception(tmp_path, monkeypatch):
    csv_path = tmp_path / "dataset.csv"

    def fake_download(url, output, quiet):
        with open(output, "w", encoding="utf-8") as file:
            file.write("partial data")

        raise ConnectionError("Network error")

    monkeypatch.setitem(
        sys.modules,
        "gdown",
        SimpleNamespace(download=fake_download)
    )

    with pytest.raises(RuntimeError, match="Could not download"):
        ensure_dataset_available(str(csv_path))

    # فایل ناقص باید حذف شده باشد
    assert not csv_path.exists()

def test_download_returns_none(tmp_path, monkeypatch):
    csv_path = tmp_path / "dataset.csv"

    mock_download = Mock(return_value=None)

    monkeypatch.setitem(
        sys.modules,
        "gdown",
        SimpleNamespace(download=mock_download)
    )

    with pytest.raises(RuntimeError, match="did not return the dataset"):
        ensure_dataset_available(str(csv_path))

    assert not csv_path.exists()

def test_download_empty_file(tmp_path, monkeypatch):
    csv_path = tmp_path / "dataset.csv"

    def fake_download(url, output, quiet):
        open(output, "w").close()
        return output

    monkeypatch.setitem(
        sys.modules,
        "gdown",
        SimpleNamespace(download=fake_download)
    )

    with pytest.raises(RuntimeError, match="did not return the dataset"):
        ensure_dataset_available(str(csv_path))

    assert not csv_path.exists()

def test_create_missing_directory(tmp_path, monkeypatch):
    csv_path = tmp_path / "data" / "files" / "dataset.csv"

    def fake_download(url, output, quiet):
        with open(output, "w", encoding="utf-8") as file:
            file.write("text\nhello")

        return output

    monkeypatch.setitem(
        sys.modules,
        "gdown",
        SimpleNamespace(download=fake_download)
    )

    result = ensure_dataset_available(str(csv_path))

    assert result == str(csv_path)
    assert csv_path.parent.exists()
    assert csv_path.exists()
