import math

import pandas as pd
import pytest

from Services.aspect_service import add_aspect_columns, aggregate_aspect_scores, find_aspects
from Services.keywords import ASPECT_KEYWORDS


@pytest.mark.parametrize("text, expected", [
    ("نوبت دیر دادند", {"waiting"}),
    ("قیمت مناسب بود", {"price"}),
    ("خیلی مهربان و صبور بود", {"behavior"}),
    ("عالی بود", {"quality"}),
])
def test_find_aspects_detects_keywords(text, expected):
    assert find_aspects(text) == expected


def test_find_aspects_matches_whole_word_only():
    # «دیروز» نباید با کلیدواژه‌ی «دیر» تطبیق داده شود
    assert find_aspects("دیروز آمدم") == set()


def test_find_aspects_skips_negated_sentence():
    assert find_aspects("هزینه‌اش گران نبود") == set()


def test_find_aspects_negation_is_per_sentence():
    # جمله‌ی اول منفی است، جمله‌ی دوم نه
    assert find_aspects("نوبت دیر نبود. عالی بود") == {"quality"}


def test_find_aspects_negation_phrase():
    assert find_aspects("عدم رضایت از درمان") == set()


def test_find_aspects_multiple_aspects_in_one_comment():
    assert find_aspects("عالی بود. نوبت دیر دادند") == {"quality", "waiting"}


def test_find_aspects_handles_non_string():
    assert find_aspects(None) == set()


def _comments():
    return pd.DataFrame({
        "doctor_id": [1, 1, 2],
        "text_norm": ["نوبت دیر دادند", "نوبت معطل شدم", "عالی بود"],
        "score_10": [4.0, 4.0, 10.0],
    })


def test_add_aspect_columns_creates_flags_and_scores():
    out = add_aspect_columns(_comments())

    assert list(out["has_waiting"]) == [1, 1, 0]
    assert out.loc[0, "waiting_score_10"] == 4.0
    assert math.isnan(out.loc[2, "waiting_score_10"])
    assert out.loc[2, "has_quality"] == 1


def test_add_aspect_columns_missing_text_raises():
    with pytest.raises(KeyError, match="Missing column"):
        add_aspect_columns(pd.DataFrame({"score_10": [1.0]}))


def test_aggregate_aspect_scores_counts_and_bayes():
    prepared = add_aspect_columns(_comments())

    out = aggregate_aspect_scores(prepared, prior_weight=10).set_index("doctor_id")

    # فقط یک دسته‌ی waiting داریم، پس میانگین کل همان ۴ است و دکتر ۱ هم ۴ می‌گیرد
    assert out.loc[1, "waiting_count"] == 2
    assert out.loc[1, "waiting_score"] == pytest.approx(4.0)
    assert out.loc[2, "waiting_count"] == 0
    assert math.isnan(out.loc[2, "waiting_score"])


def test_aggregate_aspect_scores_has_all_aspect_columns():
    prepared = add_aspect_columns(_comments())

    out = aggregate_aspect_scores(prepared)

    for aspect in ASPECT_KEYWORDS:
        assert f"{aspect}_count" in out.columns
        assert f"{aspect}_score" in out.columns


def test_aggregate_requires_add_aspect_columns_first():
    with pytest.raises(KeyError, match="Run add_aspect_columns first"):
        aggregate_aspect_scores(_comments())