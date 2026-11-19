import pandas as pd
import pytest

from Services.tfidf_service import term_scores


@pytest.fixture
def scored_comments():
    # «عالی» در ۵ کامنت با امتیاز ۱۰، «بد» در ۵ کامنت با امتیاز ۰
    good = ["عالی بود", "عالی هست", "عالی خوب", "عالی واقعا", "عالی ممنون"]
    bad = ["بد بود", "بد هست", "بد نبود", "بد داشت", "بد کرد"]
    return pd.DataFrame({
        "text_norm": good + bad,
        "score_10": [10.0] * 5 + [0.0] * 5,
    })


def test_term_scores_columns(scored_comments):
    out = term_scores(scored_comments, min_df=2)

    assert list(out.columns) == ["term", "comment_count", "mean_score", "smoothed_score", "mean_tfidf"]


def test_term_scores_counts_and_means(scored_comments):
    out = term_scores(scored_comments, min_df=2, shrink_k=0).set_index("term")

    assert out.loc["عالی", "comment_count"] == 5
    assert out.loc["عالی", "mean_score"] == pytest.approx(10.0)
    assert out.loc["بد", "mean_score"] == pytest.approx(0.0)


def test_term_scores_sorted_ascending_by_smoothed(scored_comments):
    out = term_scores(scored_comments, min_df=2)

    assert out["smoothed_score"].is_monotonic_increasing
    assert out.iloc[0]["term"] == "بد"
    assert out.iloc[-1]["term"] == "عالی"


def test_term_scores_shrinkage(scored_comments):
    # global_mean = 5، count = 5، shrink_k = 20 → (5*0 + 20*5) / 25 = 4
    out = term_scores(scored_comments, min_df=2, shrink_k=20).set_index("term")

    assert out.loc["بد", "smoothed_score"] == pytest.approx(4.0)


def test_term_scores_min_df_filters_rare_terms(scored_comments):
    out = term_scores(scored_comments, min_df=6)

    assert "عالی" not in set(out["term"])  # فقط ۵ کامنت دارد