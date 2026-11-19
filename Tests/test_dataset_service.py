import pandas as pd
import pytest

from Services import dataset_service

COMMENT_COLS = ["doctor_id", "text", "rate", "label"]  # فقط ستون‌های لازم برگردانده می‌شوند


@pytest.fixture
def no_download(monkeypatch):
    """ensure_dataset_available را طوری عوض می‌کند که فقط مسیر را برگرداند."""
    monkeypatch.setattr(dataset_service, "ensure_dataset_available", lambda path, url=None: path)


def _comments_df(**overrides):
    row = {"doctor_id": 1, "text": "عالی بود", "rate": 5.0, "label": 1,
           "date": "2024-01-01", "like": 0, "dislike": 0}
    row.update(overrides)
    return pd.DataFrame([row])


def test_load_comments_returns_required_columns(tmp_path, monkeypatch, no_download, capsys):
    path = tmp_path / "comments.csv"
    _comments_df().to_csv(path, index=False)
    monkeypatch.setattr(dataset_service, "COMMENTS_FULL_CSV_PATH", str(path))

    df = dataset_service.load_comments()

    assert list(df.columns) == COMMENT_COLS
    out = capsys.readouterr().out
    assert "Empty comments: 0" in out
    assert "Rate/label mismatches: 0" in out


def test_load_comments_reports_mismatch(tmp_path, monkeypatch, no_download, capsys):
    path = tmp_path / "comments.csv"
    _comments_df(rate=2.5, label=1).to_csv(path, index=False)  # label=1 باید rate=5 باشد
    monkeypatch.setattr(dataset_service, "COMMENTS_FULL_CSV_PATH", str(path))

    dataset_service.load_comments()

    assert "Rate/label mismatches: 1" in capsys.readouterr().out


def test_load_comments_reports_empty_text(tmp_path, monkeypatch, no_download, capsys):
    path = tmp_path / "comments.csv"
    _comments_df(text="   ").to_csv(path, index=False)
    monkeypatch.setattr(dataset_service, "COMMENTS_FULL_CSV_PATH", str(path))

    dataset_service.load_comments()

    assert "Empty comments: 1" in capsys.readouterr().out


def test_load_comments_verbose_false_is_silent(tmp_path, monkeypatch, no_download, capsys):
    path = tmp_path / "comments.csv"
    _comments_df().to_csv(path, index=False)
    monkeypatch.setattr(dataset_service, "COMMENTS_FULL_CSV_PATH", str(path))

    dataset_service.load_comments(verbose=False)

    assert capsys.readouterr().out == ""


def test_load_normalized_comments_needs_text_norm(tmp_path, monkeypatch, no_download):
    path = tmp_path / "normalized.csv"
    _comments_df().to_csv(path, index=False)  # بدون text_norm
    monkeypatch.setattr(dataset_service, "NORMALIZED_COMMENTS_CSV_PATH", str(path))

    with pytest.raises(KeyError, match="text_norm"):
        dataset_service.load_normalized_comments()


def test_load_doctors_reports_duplicates(tmp_path, monkeypatch, no_download, capsys):
    path = tmp_path / "doctors.csv"
    pd.DataFrame({
        "doctor_id": [1, 1, 2],
        "name": ["A", "A", "B"],
        "specialty": ["x", "x", "y"],
        "profile_url": ["u1", "u1", "u2"],
        "rating_score_5": [4.5, 4.5, 3.0],
    }).to_csv(path, index=False)
    monkeypatch.setattr(dataset_service, "DOCTORS_RAW_CANONICAL_CSV_PATH", str(path))

    df = dataset_service.load_doctors()

    assert len(df) == 3
    assert "Duplicate doctor records: 2" in capsys.readouterr().out