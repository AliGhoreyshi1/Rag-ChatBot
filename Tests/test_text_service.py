import pandas as pd
import pytest

from Services import text_service
from Services.text_service import normalize_column, normalize_text, split_sentences


@pytest.fixture(autouse=True)
def no_hazm(monkeypatch):
    """برای تست‌ها hazm را خاموش می‌کنیم تا نتیجه فقط به قواعد خودمان وابسته باشد."""
    monkeypatch.setattr(text_service, "_HAZM", None)


def test_elongation_is_collapsed():
    assert normalize_text("عاااالی") == "عالی"
    assert normalize_text("عالییییی") == "عالی"


def test_zwnj_becomes_space():
    assert normalize_text("سلام\u200cدکتر") == "سلام دکتر"


@pytest.mark.parametrize("raw, expected", [
    ("فوقالعاده", "فوق العاده"),
    ("بسیارعالی", "بسیار عالی"),
    ("باحوصله", "با حوصله"),
    ("درحال", "در حال"),
    ("خوشبرخورد", "خوش برخورد"),
])
def test_spacing_map_applies(raw, expected):
    assert normalize_text(raw) == expected


def test_spacing_map_respects_punctuation():
    assert normalize_text("فوقالعاده.") == "فوق العاده."


def test_spacing_map_does_not_touch_inside_longer_word():
    # «درحالی» کلمه‌ی دیگری است و نباید تغییر کند
    assert normalize_text("درحالی") == "درحالی"


def test_extra_spaces_are_removed():
    assert normalize_text("  خیلی   خوب  ") == "خیلی خوب"


@pytest.mark.parametrize("value", [None, float("nan"), 12])
def test_non_string_returns_empty(value):
    assert normalize_text(value) == ""


def test_normalize_column_adds_new_column_and_keeps_original():
    df = pd.DataFrame({"text": ["عاااالی", "درحال بهبود"]})

    out = normalize_column(df)

    assert list(out["text_norm"]) == ["عالی", "در حال بهبود"]
    assert list(out["text"]) == ["عاااالی", "درحال بهبود"]
    assert "text_norm" not in df.columns  # ورودی تغییر نکرده


def test_normalize_column_missing_column_raises():
    with pytest.raises(KeyError, match="Missing text column"):
        normalize_column(pd.DataFrame({"other": ["x"]}))


def test_split_sentences_splits_on_punctuation_and_newline():
    assert split_sentences("سلام. خوبی؟\nخداحافظ") == ["سلام", "خوبی", "خداحافظ"]


def test_split_sentences_drops_empty_parts():
    assert split_sentences("...  !!") == []


def test_split_sentences_non_string_is_empty():
    assert split_sentences(None) == []