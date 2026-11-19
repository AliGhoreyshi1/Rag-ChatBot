import math

import pandas as pd
import pytest

from Services.keywords import ASPECT_KEYWORDS
from Services.rag_service import build_doctor_frame, prepare_comments


@pytest.fixture
def doctors():
    return pd.DataFrame({
        "doctor_id": [1, 2],
        "name": ["دکتر الف", "دکتر ب"],
        "specialty": ["قلب", "پوست"],
        "profile_url": ["u1", "u2"],
        "rating_score_5": [4.5, 3.0],
        "extra_column": ["x", "y"],
    })


@pytest.fixture
def comments():
    raw = pd.DataFrame({
        "doctor_id": [1, 1],
        "text_norm": ["عالی بود", "نوبت دیر دادند"],
        "rate": [5.0, 2.5],
    })
    return prepare_comments(raw)


def test_prepare_comments_adds_score_and_aspects(comments):
    assert "score_10" in comments.columns
    assert "has_quality" in comments.columns
    assert list(comments["score_10"]) == [10.0, 5.0]


def test_build_doctor_frame_keeps_doctor_without_comments(doctors, comments):
    out = build_doctor_frame(doctors, comments).set_index("doctor_id")

    assert len(out) == 2
    assert out.loc[2, "comment_count"] == 0
    assert math.isnan(out.loc[2, "comment_rating"])
    assert out.loc[2, "waiting_count"] == 0
    assert math.isnan(out.loc[2, "waiting_score"])


def test_build_doctor_frame_program_rating_is_scaled(doctors, comments):
    out = build_doctor_frame(doctors, comments).set_index("doctor_id")

    assert out.loc[1, "program_rating"] == pytest.approx(9.0)
    assert out.loc[2, "program_rating"] == pytest.approx(6.0)


def test_build_doctor_frame_column_order(doctors, comments):
    out = build_doctor_frame(doctors, comments)

    expected_front = ["doctor_id", "name", "specialty", "profile_url",
                      "program_rating", "comment_count", "comment_rating"]
    expected_rest = [c for a in ASPECT_KEYWORDS for c in (f"{a}_count", f"{a}_score")]
    assert list(out.columns) == expected_front + expected_rest


def test_build_doctor_frame_drops_extra_doctor_columns(doctors, comments):
    out = build_doctor_frame(doctors, comments)

    assert "extra_column" not in out.columns
    assert "rating_score_5" not in out.columns


def test_build_doctor_frame_counts_are_integers(doctors, comments):
    out = build_doctor_frame(doctors, comments)

    assert out["comment_count"].dtype.kind == "i"
    assert out[[f"{a}_count" for a in ASPECT_KEYWORDS]].dtypes.apply(lambda d: d.kind == "i").all()