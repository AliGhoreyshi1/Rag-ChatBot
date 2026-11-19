import pandas as pd

from Services.plot_service import fa, plot_scatter, plot_terms


def _result():
    return pd.DataFrame({
        "term": ["بد", "عالی", "خوب"],
        "comment_count": [60, 80, 90],
        "smoothed_score": [3.0, 9.5, 8.0],
    })


def test_fa_returns_string():
    assert isinstance(fa("عالی"), str)


def test_plot_terms_writes_png(tmp_path):
    out = tmp_path / "terms.png"

    plot_terms(_result(), global_mean=8.0, out_path=str(out), top_n=2, min_count=50)

    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_terms_empty_after_filter_still_writes(tmp_path):
    out = tmp_path / "terms.png"

    plot_terms(_result(), global_mean=8.0, out_path=str(out), top_n=2, min_count=1000)

    assert out.exists()


def test_plot_scatter_writes_png(tmp_path):
    out = tmp_path / "scatter.png"

    plot_scatter(_result(), str(out))

    assert out.exists()
    assert out.stat().st_size > 0