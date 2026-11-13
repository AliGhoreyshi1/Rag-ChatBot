import re
import unicodedata
from collections import Counter
from typing import Dict, List, Tuple, Union
from Helpers.BPE_tokenizer import BPETokenizer

try:
    import regex
    GPT2_PATTERN = regex.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
except ImportError:
    GPT2_PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+"""
    )

def pre_tokenize(text: str) -> List[str]:
    """
    Split input text into initial word/symbol chunks using the GPT-2 regex pattern.

    Args:
        text (str): Raw input text string to pre-tokenize.

    Returns:
        List[str]: A list of string chunks matched by the pre-tokenization regex.
    """
    return [match.group() for match in GPT2_PATTERN.finditer(text)]

def apply_merge(byte_seq: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
    """
    Replace consecutive occurrences of a specific pair of token IDs in a sequence with a new token ID.

    Args:
        byte_seq (List[int]): Current sequence of token IDs.
        pair (Tuple[int, int]): A tuple (first_id, second_id) representing the pair to merge.
        new_id (int): The new token ID assigned to the merged pair.

    Returns:
        List[int]: A new list of token IDs with target pairs merged.
    """
    result = []
    i = 0
    while i < len(byte_seq):
        if i + 1 < len(byte_seq) and (byte_seq[i], byte_seq[i + 1]) == pair:
            result.append(new_id)
            i += 2
        else:
            result.append(byte_seq[i])
            i += 1
    return result

class SpecialTokenHandler:
    """
    Manages registration and regex-based splitting of special tokens during tokenization.
    """

    def __init__(self) -> None:
        """Initialize empty special tokens mapping and pattern compiler."""
        self.special_tokens: Dict[str, int] = {}
        self.pattern: Union[re.Pattern, None] = None

    def add_token(self, token_str: str, token_id: int) -> None:
        """
        Register a special token and update the combined regular expression pattern.

        Args:
            token_str (str): The string representation of the special token (e.g., '<|end|>').
            token_id (int): The integer vocabulary ID assigned to the special token.

        Returns:
            None
        """
        self.special_tokens[token_str] = token_id
        ordered = sorted(self.special_tokens, key=len, reverse=True)
        self.pattern = re.compile("|".join(re.escape(k) for k in ordered))

    def split_with_specials(self, text: str) -> List[Tuple[str, bool]]:
        """
        Segment text into a list of tuples containing text chunks and boolean flags indicating special tokens.

        Args:
            text (str): Input text that may contain special tokens.

        Returns:
            List[Tuple[str, bool]]: A list of tuples where each tuple is (substring, is_special_flag).
        """
        if self.pattern is None:
            return [(text, False)] if text else []

        result: List[Tuple[str, bool]] = []
        last_end = 0
        for match in self.pattern.finditer(text):
            if match.start() > last_end:
                result.append((text[last_end:match.start()], False))
            result.append((match.group(), True))
            last_end = match.end()

        if last_end < len(text):
            result.append((text[last_end:], False))
        return result

class ProductionTokenizer(BPETokenizer):
    """
    Byte-Pair Encoding (BPE) tokenizer supporting training, normalization, special tokens, encoding, and decoding.
    """

    def __init__(self) -> None:
        """Initialize vocabulary, merges, special token handler, and next available token ID."""
        super().__init__()
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.special_handler = SpecialTokenHandler()
        self.next_id: int = 256

    def normalize(self, text: str) -> str:
        """
        Normalize input text using Unicode NFKC normalization.

        Args:
            text (str): Raw input string.

        Returns:
            str: Unicode normalized string.
        """
        return unicodedata.normalize('NFKC' , text)

    def _to_byte_chunks(self, text: str) -> List[List[int]]:
        """Pre-tokenize plain text and convert each chunk to a list of byte values.

        Args:
            text (str): Text without special tokens.

        Returns:
            List[List[int]]: One byte sequence per chunk.
        """
        return [list(chunk.encode("utf-8")) for chunk in pre_tokenize(text)]
    
    def train(self, text: str, num_merges: int) -> None:
        """
        Train the BPE tokenizer by identifying frequent adjacent pairs and iteratively merging them.

        Args:
            text (str): Corpus text used for training the tokenizer.
            num_merges (int): Number of BPE merge operations to execute.

        Returns:
            None
        """
        if self.merges:
            raise RuntimeError("train() can only be called once on a fresh tokenizer.")
        
        text = self.normalize(text)
        byte_chunks = []
        for part, is_special in self.special_handler.split_with_specials(text):
            if not is_special:
                byte_chunks.extend(self._to_byte_chunks(part))

        for _ in range(num_merges):
            pair_counts = Counter(p for seq in byte_chunks for p in zip(seq, seq[1:]))
            if not pair_counts:
                break
            best_pair, _ = pair_counts.most_common(1)[0]

            new_id = self.next_id
            self.next_id += 1
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            byte_chunks = [self._merge_pair(seq, best_pair, new_id) for seq in byte_chunks]

    def add_special_token(self, token_str: str) -> int:
        """
        Register a new special token into the tokenizer vocabulary and special token handler.

        Args:
            token_str (str): Special token string (e.g., '<|begin|>').

        Returns:
            int: Assigned vocabulary integer ID for the special token.
        """
        new_id = self.next_id
        self.next_id += 1
        self.special_handler.add_token(token_str, new_id)
        self.vocab[new_id] = token_str.encode("utf-8")
        return new_id

    def encode(self, text: str) -> List[int]:
        """
        Encode raw text into a sequence of vocabulary token IDs using trained merges and special tokens.

        Args:
            text (str): Input text string to be tokenized.

        Returns:
            List[int]: List of encoded vocabulary token IDs.
        """
        text = self.normalize(text)
        ids: List[int] = []
        for part, is_special in self.special_handler.split_with_specials(text):
            if is_special:
                ids.append(self.special_handler.special_tokens[part])
                continue
            for seq in self._to_byte_chunks(part):
                for pair, new_id in self.merges.items():
                    seq = self._merge_pair(seq, pair, new_id)
                ids.extend(seq)
        return ids

    def decode(self, ids: List[int]) -> str:
        """
        Decode a list of token IDs back into a UTF-8 string.

        Args:
            ids (List[int]): List of integer token IDs.

        Returns:
            str: Decoded UTF-8 text string.
        """
        return b"".join(self.get_token_bytes(i) for i in ids).decode("utf-8", errors="replace")

    def get_token_bytes(self, token_id: int) -> bytes:
        """
        Retrieve underlying byte representation of a given token ID.

        Args:
            token_id (int): Token ID to look up.

        Returns:
            bytes: Byte sequence corresponding to token_id, or default placeholder if not found.
        """
        try:
            return self.vocab[token_id]
        except KeyError as e:
            raise KeyError(f"Unknown token id: {token_id}") from e