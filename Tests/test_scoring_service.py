import math

import pandas as pd
import pytest

from Services.scoring_service import add_score_10, bayes_mean, overall_comment_scores


def test_add_score_10_doubles_rate():
    df = pd.DataFrame({"rate": [0.0, 2.5, 5.0]})

    out = add_score_10(df)

    assert list(out["score_10"]) == [0.0, 5.0, 10.0]
    assert "score_10" not in df.columns  # ورودی تغییر نکرده


def test_add_score_10_missing_column_raises():
    with pytest.raises(KeyError, match="Missing column"):
        add_score_10(pd.DataFrame({"other": [1]}))


def test_bayes_mean_with_zero_count_returns_global_mean():
    assert bayes_mean(0, 999.0, global_mean=7.0, prior_weight=10) == pytest.approx(7.0)


def test_bayes_mean_formula():
    # (10*6 + 10*8) / 20 = 7
    assert bayes_mean(10, 8.0, global_mean=6.0, prior_weight=10) == pytest.approx(7.0)


def test_bayes_mean_large_count_approaches_doctor_mean():
    value = bayes_mean(10_000, 8.0, global_mean=6.0, prior_weight=10)
    assert value == pytest.approx(8.0, abs=0.01)


def test_overall_comment_scores_shrinks_small_sample():
    df = pd.DataFrame({
        "doctor_id": [1, 1, 2] + [2] * 9,
        "score_10": [10.0, 10.0] + [4.0] * 10,
    })
    global_mean = df["score_10"].mean()

    out = overall_comment_scores(df, prior_weight=10).set_index("doctor_id")

    expected_doctor1 = (10 * global_mean + 2 * 10.0) / (10 + 2)
    assert out.loc[1, "comment_count"] == 2
    assert out.loc[1, "comment_rating"] == pytest.approx(round(expected_doctor1, 2))
    # دکتر با تعداد کامنت کم نباید از دکتر با کامنت زیاد بالاتر شود
    assert out.loc[1, "comment_rating"] < 10.0


def test_overall_comment_scores_output_columns():
    df = pd.DataFrame({"doctor_id": [1], "score_10": [5.0]})

    out = overall_comment_scores(df)

    assert list(out.columns) == ["doctor_id", "comment_count", "comment_rating"]
    assert not math.isnan(out.loc[0, "comment_rating"])