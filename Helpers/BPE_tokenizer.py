from collections import Counter
from typing import List, Dict, Tuple, Any, Optional


class CharTokenizer:
    """A simple character-level tokenizer mapping ASCII/Unicode characters to integer values."""

    def encode(self, text: str) -> List[int]:
        """Convert an input string into a list of integer character codes.
        

        Args:
            text (str): Input text to encode.

        Returns:
            List[int]: List of character integer codes.
        """
        try: 
            extracted_letters = list(text)

        except TypeError as e:
            raise TypeError("text must be a string.") from e

        tokens = []

        for char in extracted_letters:
            try:
                tokens.append(ord(char))

            except TypeError as e:
                raise ValueError(f"Invalid character: {char!r}") from e
        
        return tokens

    def decode(self, tokens: List[int]) -> str:
        """Convert a list of integer character codes back into a string.

        Args:
            tokens (List[int]): List of integer character codes.

        Returns:
            str: Decoded text string.
        """
        if not isinstance(tokens, list):
            raise TypeError("tokens must be a list of integer.")
    
        text = ""
        char_list = []
    
        for token in tokens:
            if isinstance(token, bool) or not isinstance(token, int):
                raise TypeError(
                    f"Each token must be an integer. Got {token!r}"
                )

            try:
                char_list.append(chr(token))

            except ValueError as e:
                raise ValueError(
                    f"Invalid character code: {token!r}"
                ) from e
        text = ''.join(char_list)
        return text

class BPETokenizer:
    """Byte-Pair Encoding (BPE) Tokenizer implementation starting from 256 base byte tokens."""

    def __init__(self) -> None:
        """Initialize vocabulary and merge rules storage."""
        self.merges: Dict[Tuple[int, int], int] = {}
        self.vocab: Dict[int, bytes] = {}

    def _get_pairs(self, tokens: List[int]) -> Counter:
        """Count frequencies of adjacent token pairs in a sequence.

        Args:
            tokens (List[int]): List of current integer tokens. Shape: (seq_len,)

        Returns:
            Counter: Mapping of (token_a, token_b) tuples to their occurrence counts.
        """
        try:            
            Mapping_of_tuples = list(zip(tokens, tokens[1:]))
        
            result = Counter(Mapping_of_tuples)
        
            return result

        except TypeError as e:
            raise TypeError(
                "tokens must contain hashable elements."
            ) from e

    def _merge_pair(
        self, tokens: List[int], pair: Tuple[int, int], new_token: int
    ) -> List[int]:
        """Replace non-overlapping occurrences of a target adjacent pair with a new merged token ID.

        Args:
            tokens (List[int]): Current token sequence. Shape: (seq_len,)
            pair (Tuple[int, int]): Target pair of token IDs to replace.
            new_token (int): New token ID to insert for the target pair.

        Returns:
            List[int]: Updated sequence with non-overlapping target pair merged.
        """
        pairs = self._get_pairs(tokens)

        if pair not in pairs:
            return tokens

        new_list = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and (tokens[i], tokens[i+1]) == pair:
                new_list.append(new_token)
                i += 2

            else:
                new_list.append(tokens[i])
                i += 1

        return new_list

    def train(self, text: str, num_merges: int) -> "BPETokenizer":
        """Train BPE vocabulary starting from 256 base bytes and learn merge rules from text.

        Args:
            text (str): Raw text corpus for training.
            num_merges (int): Total number of BPE merge operations to execute.

        Returns:
            BPETokenizer: Self instance after training.
        """
        if not self.vocab:
            for i in range(256):
                self.vocab[i] = bytes([i])
        
        tokens = list(text.encode("utf-8"))
        new_token = 256

        for _ in range(num_merges):
            pairs = self._get_pairs(tokens)
            if not pairs:
                return self
            
            pair, frequency = pairs.most_common(1)[0]
            pair1, pair2 = pair

            vocab1 = self.vocab[pair1]
            vocab2 = self.vocab[pair2]

            self.vocab[new_token] = vocab1+vocab2
            self.merges[pair] = new_token

            tokens = self._merge_pair(tokens, pair, new_token)
            new_token += 1

        return self

    def encode(self, text: str) -> List[int]:
        """Encode string text into BPE token IDs using learned merge rules.

        Args:
            text (str): Input text string.

        Returns:
            List[int]: Encoded list of BPE token IDs.
        """
        try:
            tokens = list(text.encode("utf-8"))
        
            for pair, new_token in self.merges.items():
                tokens = self._merge_pair(tokens, pair, new_token)
        
        except TypeError as e:
            raise TypeError("text has error.") from e
    
        return tokens

    def decode(self, tokens: List[int]) -> str:
        """Decode a list of BPE token IDs back into a text string.

        Args:
            tokens (List[int]): List of BPE token IDs.

        Returns:
            str: Reconstructed text string.
        """
        try:
            text = ""
            for token in tokens:
                text+=self.token_to_str(token)

        except TypeError as e:
            raise TypeError("tokens must be an iterable of integers.") from e

        except (ValueError, UnicodeDecodeError) as e:
            raise ValueError("tokens must contain valid UTF-8 bytes.") from e

        return text

    def vocab_size(self) -> int:
        """Get current vocabulary size.

        Returns:
            int: Total number of unique tokens in the vocabulary.
        """
        return len(self.vocab) 

    def token_to_str(self, token_id: int) -> str:
        """Convert a single token ID into its string representation for visualization.

        Args:
            token_id (int): Token ID to inspect.

        Returns:
            str: String representation of the token bytes.
        """
        try:
            if token_id in self.vocab:
                return self.vocab[token_id].decode("utf-8")

            elif 0 <= token_id <= 255:
                return bytes([token_id]).decode("utf-8")

            else:
                return f"<unk:{token_id}>"

        except TypeError as e:
            raise TypeError("tokens must be an iterable of integers.") from e

def compression_ratio(tokenizer: Any, text: str) -> float:
    """Calculate token compression ratio relative to raw UTF-8 byte count.

    Handle empty strings gracefully by returning 0.0 when byte length is zero.

    Args:
        tokenizer (Any): Tokenizer instance providing encode().
        text (str): Input text string.

    Returns:
        float: Ratio of encoded token count to raw byte length (0.0 if empty).
    """
    byte_length = len(text.encode("utf-8"))

    if not byte_length:
        return 0.0

    tokens = tokenizer.encode(text)
    token_count = len(tokens)

    ratio = token_count / byte_length

    return ratio

def vocabulary_stats(tokenizer: Any, texts: List[str]) -> None:
    """Compute and display usage statistics across a list of text corpora.

    Args:
        tokenizer (Any): Tokenizer instance providing encode(), vocab_size(), and token_to_str().
        texts (List[str]): List of text strings to analyze.

    Returns:
        None
    """
    all_tokens = []
    all_bytes = []

    for i, text in enumerate(texts, 1):
        encoded = tokenizer.encode(text)
        byte_count = len(text.encode("utf-8"))

        all_tokens.extend(encoded)
        all_bytes.append(byte_count)

        print(f"Text {i}: compression = {len(encoded) / byte_count:.3f}")

    total_tokens = len(all_tokens)
    vocab_size = tokenizer.vocab_size()
    
    print(f"Total tokens:    {total_tokens}")
    print(f"Vocab size:      {vocab_size}")
    print(f"Unique tokens:   {len(set(all_tokens))} / {vocab_size}")
    
    token_freq = Counter(all_tokens)
    most_common = token_freq.most_common(3)
    
    for token_id, count in most_common:
        print(f"Token {token_id}: {count} times")