
import pickle
from unittest.mock import Mock

import Services.model_train as model_train


class DummyTokenizer:
    """A lightweight tokenizer used for testing."""

    def vocab_size(self):
        return 300

    def encode(self, text):
        return [ord(char) for char in text]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


def test_bpe_on_data_success(tmp_path, monkeypatch, capsys):
    # Arrange
    csv_path = str(tmp_path / "comments.csv")

    docs = [
        "This is a sample document for testing BPE tokenizer."
    ]

    final_docs = [
        "This is a sample document for testing BPE tokenizer."
    ]

    packed = {
        "input_ids": [[1, 2, 3], [4, 5, 6]]
    }

    train = {
        "input_ids": [[1, 2, 3]]
    }

    val = {
        "input_ids": [[4, 5, 6]]
    }

    tokenizer = DummyTokenizer()

    mock_comments = Mock(return_value=docs)
    mock_build = Mock(
        return_value=(tokenizer, packed, final_docs)
    )
    mock_split = Mock(
        return_value=(train, val)
    )

    monkeypatch.setattr(
        model_train, "comments_to_documents", mock_comments
    )
    monkeypatch.setattr(
        model_train, "build_training_data", mock_build
    )
    monkeypatch.setattr(
        model_train, "split_packed", mock_split
    )

    monkeypatch.setattr(model_train, "NUM_MERGES", 100)
    monkeypatch.setattr(model_train, "SEQ_LENGTH", 128)

    monkeypatch.chdir(tmp_path)

    # Act
    model_train.bpe_on_data(csv_path)

    # Assert: function calls
    mock_comments.assert_called_once_with(
        csv_path,
        text_column="text",
        group_column="doctor_id",
        min_words=60
    )

    mock_build.assert_called_once_with(
        docs,
        num_merges=100,
        seq_length=128
    )

    mock_split.assert_called_once_with(
        packed,
        val_fraction=0.15
    )

    # Assert: tokenizer roundtrip
    sample = final_docs[0][:200]
    assert tokenizer.decode(tokenizer.encode(sample)) == sample

    # Assert: generated files
    tokenizer_path = tmp_path / "tokenizer_100.pkl"
    packed_path = tmp_path / "packed_100.pkl"

    assert tokenizer_path.exists()
    assert packed_path.exists()

    # Assert: saved tokenizer
    with open(tokenizer_path, "rb") as file:
        saved_tokenizer = pickle.load(file)

    assert saved_tokenizer.vocab_size() == 300

    # Assert: saved training data
    with open(packed_path, "rb") as file:
        saved_data = pickle.load(file)

    assert saved_data == {
        "train": train,
        "val": val
    }

    # Assert: console output
    captured = capsys.readouterr()

    assert "Documents from comments: 1" in captured.out
    assert "Final documents: 1" in captured.out
    assert "Vocab size: 300" in captured.out
    assert "Training blocks: 2" in captured.out
    assert "Train blocks: 1, Val blocks: 1" in captured.out
    assert "Roundtrip OK: True" in captured.out
    assert "Saved tokenizer_100.pkl" in captured.out
