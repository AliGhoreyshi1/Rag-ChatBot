
import pandas as pd
import pytest

from Services import model_train


# ۱. تست بارگذاری موفق دیتاست
def test_load_dataset_success(tmp_path):
    csv_path = tmp_path / "dataset.csv"

    df = pd.DataFrame({
        "name": ["Ali", "Reza", "Ali"],
        "text": ["Hello", "Hi", "How are you?"]
    })

    df.to_csv(csv_path, index=False)

    result = model_train.load_dataset(str(csv_path))

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert len(result) == 3
    assert list(result.columns) == ["name", "text"]


# ۲. تست نبودن ستون name
def test_missing_required_column(tmp_path):
    csv_path = tmp_path / "dataset.csv"

    df = pd.DataFrame({
        "username": ["Ali", "Reza"],
        "text": ["Hello", "Hi"]
    })

    df.to_csv(csv_path, index=False)

    with pytest.raises(
        KeyError,
        match="Required column not found"
    ):
        model_train.load_dataset(str(csv_path))


# ۳. تست فایل CSV خالی
def test_empty_csv(tmp_path):
    csv_path = tmp_path / "dataset.csv"
    csv_path.write_text("", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="Invalid value encountered"
    ):
        model_train.load_dataset(str(csv_path))


# ۴. تست مسیر نامعتبر فایل
def test_file_not_found(tmp_path):
    csv_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        model_train.load_dataset(str(csv_path))


# ۵. تست خطاهای اختصاصی تابع
@pytest.mark.parametrize(
    "exception, expected_message",
    [
        (IndexError, "Index is out of range"),
        (ValueError, "Invalid value encountered"),
        (TypeError, "Invalid data type provided"),
        (AttributeError, "An attribute error occurred"),
    ]
)
def test_exception_handling(
    monkeypatch,
    exception,
    expected_message
):
    def mock_read_csv(*args, **kwargs):
        raise exception("Test error")

    monkeypatch.setattr(
        model_train.pd,
        "read_csv",
        mock_read_csv
    )

    with pytest.raises(
        exception,
        match=expected_message
    ) as exc_info:
        model_train.load_dataset("fake.csv")

    # بررسی حفظ Exception اصلی
    assert isinstance(exc_info.value.__cause__, exception)
