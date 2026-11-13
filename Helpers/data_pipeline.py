import re
import csv
import hashlib
import random
import time
from collections import Counter, defaultdict
from typing import Iterable, List, Tuple, Set, Dict, Any, Generator, Optional
from Helpers.production_tokenizer import ProductionTokenizer
from hazm import Normalizer

# Persian normalizer from hazm. persian_numbers=False keeps digits as they are, so
# the number handling in remove_pii() and _PERSIAN_DIGITS still sees the original text.
_HAZM_NORMALIZER = Normalizer(persian_numbers=False)


def normalize_text(text: str) -> str:
    """
    Persian text normalization with hazm.

    Converts Arabic yeh/kaf to Persian ی/ک, removes tatweel and diacritics,
    fixes spacing around punctuation and affixes, and standardizes the zero-width
    non-joiner (ZWNJ). Digits are left unchanged.

    Args:
        text (str): Input text.

    Returns:
        str: Normalized text.
    """
    return _HAZM_NORMALIZER.normalize(text)

def clean_text(text: str) -> str:
    """
    Cleans raw document text: strips URLs and HTML tags, normalizes Persian text with
    hazm, and collapses whitespace.

    Order matters: URLs and tags are removed first so that hazm only sees visible text.
    Unicode NFKC normalization is still applied by the tokenizer on top of this.

    Args:
        text (str): Input raw text document.

    Returns:
        str: Cleaned and normalized text string.
    """
    text = re.sub(r"(?:https?://|www\.)[^\s<>\"']+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]*>", " ", text)
    text = normalize_text(text)
    return re.sub(r"\s+", " ", text).strip()

def quality_filter(
    text: str, 
    min_words: int = 50, 
    max_ratio_caps: int = 0.3, 
    max_ratio_special: float = 0.1
) -> bool:
    """
        Filters out low-quality documents based on length, capitalization ratio, and special character density.
    
        Length is counted with _units(), so scripts without spaces (CJK) are measured
        in characters rather than being treated as one giant word.
    
        Args:
            text (str): Cleaned document text.
            min_words (int): Minimum required unit count (words, or CJK characters).
            max_ratio_caps (float): Maximum allowed ratio of ALL-CAPS words.
            max_ratio_special (float): Maximum allowed ratio of non-alphanumeric special characters.
    
        Returns:
            bool: True if the document meets quality criteria, False otherwise.
        """
    words = text.split()
    if not words or len(_units(text)) < min_words:
        return False

    caps_ratio = sum(word.isupper() for word in words) / len(words)
    if caps_ratio > max_ratio_caps:
        return False

    special_count = sum(not ch.isalnum() and not ch.isspace() for ch in text)
    special_ratio = special_count / len(text) if text else 0.0
    return special_ratio <= max_ratio_special

# CJK scripts (kana, ideographs, hangul) are written without spaces, so each
# character is treated as its own unit. Other scripts use whitespace-free word runs.
_CJK_RANGES = "\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af"
_UNIT_PATTERN = re.compile(rf"[{_CJK_RANGES}]|[^\W{_CJK_RANGES}]+")

def _units(text: str) -> List[str]:
    """
    Split text into lowercase units: word runs for most scripts, single characters for CJK.

    Args:
        text (str): Input text.

    Returns:
        List[str]: Lowercased units in order of appearance.
    """
    return _UNIT_PATTERN.findall(text.lower())

def get_shingles(text: str, k: int = 5) -> Set[str]:
    """
    Extracts k-shingles (word n-grams) from a text string.

    Args:
        text (str): Input document text.
        k (int): Size of word shingle (n-gram length).

    Returns:
        Set[str]: Unique set of k-word shingles.
    """
    if k<=0:
        raise ValueError("k must be positive")
    words = re.findall(r"\b\w+\b", text.lower())
    return {" ".join(words[i:i+k]) for i in range(len(words)- k + 1)}

def minhash_signature(shingles: Set[str], num_hashes: int = 128) -> List[int]:
    """
    Computes MinHash signature for a set of shingles using hash function permutations.

    Args:
        shingles (Set[str]): Set of unique word shingles.
        num_hashes (int): Length of the MinHash signature vector.

    Returns:
        List[int]: MinHash signature list of length (num_hashes,).
    """
    if num_hashes<0:
        raise ValueError("num_hashes must be nonnegetive")
    if not shingles:
        return [0]*num_hashes

    signature = []
    encoded_shingles = [shingle.encode("utf-8") for shingle in shingles]
    for seed in range(num_hashes):
        prefix = seed.to_bytes(8, byteorder="big")
        minimum = min(
            int.from_bytes(hashlib.sha256(prefix+shingle).digest()[:8], "big")
            for shingle in encoded_shingles
        )
        signature.append(minimum)
    return signature

def lsh_buckets(signature: List[int], bands: int = 16) -> List[Tuple[int, str]]:
    """
    Divides MinHash signature into bands and maps each band to a bucket identifier using LSH.

    Args:
        signature (List[int]): MinHash signature vector of shape (num_hashes,).
        bands (int): Number of bands to divide the signature into.

    Returns:
        List[Tuple[int, str]]: List of (band_id, bucket_hash) tuples of length (bands,).
    """
    if bands <= 0:
            raise ValueError("bands must be a positive integer")
    if len(signature) < bands:
        raise ValueError("signature must contain at least one value per band")

    buckets = []
    for band_id in range(bands):
        start = band_id * len(signature) // bands
        end = (band_id + 1) * len(signature) // bands
        chunk = signature[start:end]
        chunk_bytes = ",".join(str(value) for value in chunk).encode("ascii")
        buckets.append((band_id, hashlib.sha256(chunk_bytes).hexdigest()))
    return buckets

def deduplicate(
    documents: List[str], 
    threshold: float = 0.8, 
    num_hashes: int = 128, 
    bands: int = 16
) -> Tuple[List[str], int]:
    """
    Finds and removes near-duplicate documents using MinHash LSH and Jaccard similarity evaluation.

    Args:
        documents (List[str]): List of clean document strings.
        threshold (float): Jaccard similarity threshold for considering two docs as duplicates.
        num_hashes (int): Total hash functions for MinHash computation.
        bands (int): Number of bands for LSH partitioning.

    Returns:
        Tuple[List[str], int]: Tuple containing (list of deduplicated documents, count of removed duplicates).
    """
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")
    if num_hashes < bands or bands <= 0:
        raise ValueError("num_hashes must be >= bands > 0")
    if len(documents) < 2:
        return list(documents), 0

    shingle_sets = [get_shingles(document, k=5) for document in documents]
    signatures = [minhash_signature(shingle, num_hashes) for shingle in shingle_sets]

    buckets = defaultdict(list)
    for doc_id, signature in enumerate(signatures):
        for bucket in lsh_buckets(signature, bands):
            buckets[bucket].append(doc_id)

    candidate_pairs = set()
    for indices in buckets.values():
        for position, left in enumerate(indices):
            for right in indices[position + 1:]:
                candidate_pairs.add((left, right))  

    buckets = defaultdict(list)
    for doc_id, signature in enumerate(signatures):
        for bucket in lsh_buckets(signature, bands):
            buckets[bucket].append(doc_id)

    candidate_pairs = set()
    for indices in buckets.values():
        for position, left in enumerate(indices):
            for right in indices[position + 1:]:
                candidate_pairs.add((left, right))

    removed = set()
    # Keep the earliest occurrence of each duplicate, retaining original order.
    for left, right in sorted(candidate_pairs):
        if left in removed or right in removed:
            continue
        union = shingle_sets[left] | shingle_sets[right]
        similarity = (len(shingle_sets[left] & shingle_sets[right]) / len(union)
                    if union else float(documents[left] == documents[right]))
        if similarity >= threshold:
            removed.add(right)

    deduplicated = [doc for i, doc in enumerate(documents) if i not in removed]
    return deduplicated, len(removed)

class SimpleTokenizer:
    """
    A minimal Byte Pair Encoding (BPE) tokenizer implementation operating over byte sequences.
    """

    EOS_TOKEN = "<|endoftext|>"
    PAD_TOKEN = "<|pad|>"

    def __init__(self, vocab_size: int = 256):
        """
        Initializes tokenizer vocabulary and internal mapping structures.

        Args:
            vocab_size (int): Target vocabulary capacity.
        """
        self._tok = ProductionTokenizer()
        self.eos_id: int = self._tok.add_special_token(self.EOS_TOKEN)
        self.pad_id: int = self._tok.add_special_token(self.PAD_TOKEN)
        self._special_ids = {self.eos_id, self.pad_id}

    def train_bpe(self, text: str, num_merges: int) -> None:
        """
        Iteratively finds the most frequent pair of tokens and merges them into a new token.

        Args:
            text (str): Training text corpus.
            num_merges (int): Number of BPE merge operations to execute.

        Returns:
            None
        """
        self._tok.train(text, num_merges)

    def encode(self, text: str) -> List[int]:
        """
        Encodes input text into a list of token IDs using pre-learned BPE merge rules.

        Args:
            text (str): Input text string.

        Returns:
            List[int]: Encoded list of token IDs.
        """
        return self._tok.encode(text)

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decodes a list of token IDs back into a UTF-8 string.

        Args:
            ids (List[int]): List of token IDs.

        Returns:
            str: Decoded text representation.
        """
        if skip_special_tokens:
            ids = [i for i in ids if i not in self._special_ids]
        return self._tok.decode(ids)

    def vocab_size(self) -> int:
        """
        Returns the total vocabulary size including base bytes and merges.

        Returns:
            int: Total vocabulary size.
        """
        return self._tok.vocab_size()

def tokenize_corpus(documents: List[str], tokenizer: SimpleTokenizer) -> List[int]:
    """
    Tokenizes a list of documents into a single flat stream of token IDs separated by EOS markers.

    Args:
        documents (List[str]): List of document strings.
        tokenizer (SimpleTokenizer): Trained BPE tokenizer instance.

    Returns:
        List[int]: Flat list containing combined token sequence.
    """
    if tokenizer.eos_id is None:
        raise ValueError("tokenizer must define an eos_id")
    token_ids: List[int] = []
    for doc in documents:
        token_ids.extend(tokenizer.encode(doc))
        token_ids.append(tokenizer.eos_id)
    return token_ids

def pack_sequences(
    token_ids: List[int], 
    seq_length: int, 
    pad_id: int = 0
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Packs continuous token stream into fixed-length sequence chunks with corresponding attention masks.

    Args:
        token_ids (List[int]): Continuous flat list of token IDs.
        seq_length (int): Target length per sequence block.
        pad_id (int): Token ID used for padding incomplete trailing sequences.

    Returns:
        Tuple[List[List[int]], List[List[int]]]: Tuple of (padded_sequences, attention_masks)
            where each element has dynamic batch outer dimension and sequence shape (seq_length,).
    """
    if seq_length <= 0:
        raise ValueError("seq_length must be positive")

    sequences: List[List[int]] = []
    masks: List[List[int]] = []
    for start in range(0, len(token_ids), seq_length):
        chunk = token_ids[start:start + seq_length]
        padding = seq_length - len(chunk)
        sequences.append(chunk + [pad_id] * padding)
        masks.append([1] * len(chunk) + [0] * padding)
    return sequences, masks

class PreTrainingDataLoader:
    """
    Data loader providing mini-batch iteration and shuffling over packed token sequences.
    """
    def __init__(
        self, 
        sequences: List[List[int]], 
        attention_masks: List[List[int]], 
        batch_size: int, 
        shuffle: bool = True,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
    ):
        """
        Initializes dataloader properties.

        Args:
            sequences (List[List[int]]): Packed input sequences of shape (num_samples, seq_length).
            attention_masks (List[List[int]]): Attention masks of shape (num_samples, seq_length).
            batch_size (int): Size of individual mini-batches.
            shuffle (bool): Whether to shuffle sample indices prior to batching.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("require 0 <= rank < world_size")
        self.sequences = sequences
        self.attention_masks = attention_masks
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
            """Set the epoch number so that each epoch gets a different, reproducible shuffle."""
            self.epoch = epoch

    def _indices(self) -> List[int]:
        indices = list(range(self.rank, len(self.sequences), self.world_size))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(indices)
        return indices

    def __len__(self) -> int:
        """
        Calculates total number of batches per epoch.

        Returns:
            int: Number of available mini-batches.
        """
        n = len(range(self.rank, len(self.sequences), self.world_size))
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Generator[Tuple[List[List[int]], List[List[int]]], None, None]:
        """
        Iterates over the dataset and yields mini-batches of sequences and attention masks.

        Returns:
            Generator[Tuple[List[List[int]], List[List[int]]], None, None]: Batch generator yielding 
            (batch_sequences, batch_attention_masks) where each tensor component has shape (batch_size, seq_length).
        """
        indices = self._indices()
        for start in range(0, len(indices), self.batch_size):
            batch = indices[start:start + self.batch_size]
            yield ([self.sequences[i] for i in batch],
                   [self.attention_masks[i] for i in batch])

# ---------------------------------------------------------------------------
# Language detection (script-based heuristic)
# ---------------------------------------------------------------------------

PERSIAN_ONLY_LETTERS = set("پچژگکی")  # letters that Arabic script does not use
_PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")

def detect_language(text: str) -> Tuple[str, float]:
    """
    Guess the dominant script/language of a text from the Unicode blocks of its letters.

    Latin-script text is labelled 'latin' (English and other Latin languages are not
    separated by this heuristic). Persian is separated from Arabic by its letters.

    Args:
        text (str): Input text.

    Returns:
        Tuple[str, float]: (language label, share of letters in that label). ('und', 0.0) if no letters.
    """
    counts: Counter = Counter()
    persian_letters = 0
    for ch in text:
        if not ch.isalpha():
            continue
        cp = ord(ch)
        if 0x3040 <= cp <= 0x30FF:
            counts["ja"] += 1
        elif 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
            counts["zh"] += 1
        elif 0xAC00 <= cp <= 0xD7AF:
            counts["ko"] += 1
        elif 0x0600 <= cp <= 0x06FF:
            counts["ar"] += 1
            if ch in PERSIAN_ONLY_LETTERS:
                persian_letters += 1
        elif cp < 0x0250:  # ASCII and Latin-1/Latin Extended
            counts["latin"] += 1
        else:
            counts["other"] += 1

    if persian_letters > 0 and counts["ar"] > 0:
        counts["fa"] = counts.pop("ar")
    total = sum(counts.values())
    if total == 0:
        return "und", 0.0
    label, top = counts.most_common(1)[0]
    return label, top / total


def language_filter(
    text: str,
    allowed: Tuple[str, ...] = ("latin", "fa", "ar", "zh", "ja", "ko"),
    min_confidence: float = 0.7
) -> bool:
    """
    Keep a document only if its dominant language is allowed and sufficiently dominant.

    Args:
        text (str): Cleaned document text.
        allowed (Tuple[str, ...]): Accepted language labels from detect_language.
        min_confidence (float): Minimum share of letters that must belong to the dominant label.

    Returns:
        bool: True if the document is kept.
    """
    label, confidence = detect_language(text)
    return label in allowed and confidence >= min_confidence

# ---------------------------------------------------------------------------
# Repetition filter and PII removal
# ---------------------------------------------------------------------------

def repeated_ngram_ratio(text: str, n: int = 3) -> float:
    """
    Fraction of unit n-grams that are repeats of an earlier n-gram (0 = no repetition).

    Args:
        text (str): Input text.
        n (int): n-gram size in units.

    Returns:
        float: Repetition ratio in [0, 1].
    """
    units = _units(text)
    grams = [tuple(units[i:i + n]) for i in range(len(units) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def is_repetitive(text: str, max_ratio: float = 0.5, n: int = 3) -> bool:
    """
    Flag documents dominated by repeated n-grams (typical of spam and keyword stuffing).

    Args:
        text (str): Cleaned document text.
        max_ratio (float): Maximum tolerated repetition ratio.
        n (int): n-gram size in units.

    Returns:
        bool: True if the document should be removed.
    """
    return repeated_ngram_ratio(text, n) > max_ratio


_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_NATIONAL_ID_RE = re.compile(r"(?<!\d)\d{10}(?!\d)")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s-]?)?\(?\d{2,4}\)?[\s-]?\d{3}[\s-]?\d{3,4}(?!\d)")


def _replace_spans(text: str, pattern: "re.Pattern", token: str) -> str:
    """Replace regex matches in `text` with `token`, matching on a digit-normalized copy."""
    normalized = text.translate(_PERSIAN_DIGITS)  # 1:1 mapping, so spans line up with `text`
    out, last = [], 0
    for m in pattern.finditer(normalized):
        out.append(text[last:m.start()])
        out.append(token)
        last = m.end()
    out.append(text[last:])
    return "".join(out)


def remove_pii(text: str) -> str:
    """
    Replace e-mail addresses, national ID numbers and phone numbers with placeholders.
    Also strips literal special-token strings so that data cannot inject EOS/PAD ids.

    Args:
        text (str): Document text.

    Returns:
        str: Text with PII replaced by <EMAIL>, <ID> and <PHONE>.
    """
    text = text.replace("<|endoftext|>", " ").replace("<|pad|>", " ")
    text = _replace_spans(text, _EMAIL_RE, "<EMAIL>")
    text = _replace_spans(text, _NATIONAL_ID_RE, "<ID>")
    text = _replace_spans(text, _PHONE_RE, "<PHONE>")
    return text

# ---------------------------------------------------------------------------
# Streaming stages
# ---------------------------------------------------------------------------

def stream_clean(raw_docs: Iterable[str], min_confidence: float = 0.7) -> Generator[str, None, None]:
    """
    Lazily apply HTML/URL cleaning, language filtering and quality filtering.

    Args:
        raw_docs (Iterable[str]): Raw documents (any iterable, e.g. a file reader).
        min_confidence (float): Passed to language_filter.

    Yields:
        str: Cleaned documents that passed the filters.
    """
    for raw in raw_docs:
        doc = clean_text(raw)
        if not language_filter(doc, min_confidence=min_confidence):
            continue
        if not quality_filter(doc) or is_repetitive(doc):
            continue
        yield doc

def stream_dedup(docs: Iterable[str], batch_size: int = 1000, threshold: float = 0.8) -> Generator[str, None, None]:
    """
    Deduplicate a stream in fixed-size batches.

    Limitation: duplicates across batches are not detected. A full-corpus deduplication
    needs a sharded index (for example, one LSH index per shard, merged in a second pass).

    Args:
        docs (Iterable[str]): Cleaned documents.
        batch_size (int): Number of documents held in memory at once.
        threshold (float): Jaccard threshold passed to deduplicate.

    Yields:
        str: Documents that survived batch-level deduplication.
    """
    batch: List[str] = []
    for doc in docs:
        batch.append(doc)
        if len(batch) == batch_size:
            kept, _ = deduplicate(batch, threshold=threshold)
            yield from kept
            batch = []
    if batch:
        kept, _ = deduplicate(batch, threshold=threshold)
        yield from kept


# ---------------------------------------------------------------------------
# Packing with document boundaries and next-token labels
# ---------------------------------------------------------------------------

def pack_documents(
    token_docs: List[List[int]],
    eos_id: int,
    seq_length: int,
    pad_id: int,
    ignore_index: int = -100,
) -> Dict[str, List[List[int]]]:
    """
    Pack tokenized documents into fixed-length blocks with per-token document ids.

    Every document ends with EOS. Labels are next-token targets: a position is supervised
    only when the next token belongs to the same document, so no loss is computed on
    predicting the first token of an unrelated document. Padding has label `ignore_index`.
    doc_ids are used by the model to build a block-diagonal attention mask.

    Args:
        token_docs (List[List[int]]): Token ids per document (without EOS).
        eos_id (int): End-of-sequence token id appended to each document.
        seq_length (int): Block length.
        pad_id (int): Padding token id.
        ignore_index (int): Label value that is excluded from the loss.

    Returns:
        Dict[str, List[List[int]]]: keys 'input_ids', 'labels', 'doc_ids', 'attention_mask'.
    """
    if seq_length < 2:
        raise ValueError("seq_length must be at least 2")

    flat_tokens: List[int] = []
    flat_docs: List[int] = []
    for doc_idx, doc in enumerate(token_docs):
        for t in doc + [eos_id]:
            flat_tokens.append(t)
            flat_docs.append(doc_idx)

    out: Dict[str, List[List[int]]] = {"input_ids": [], "labels": [], "doc_ids": [], "attention_mask": []}
    for start in range(0, len(flat_tokens), seq_length):
        toks = flat_tokens[start:start + seq_length]
        docs = flat_docs[start:start + seq_length]
        n = len(toks)
        labels = []
        for i in range(n):
            if i + 1 < n and docs[i + 1] == docs[i]:
                labels.append(toks[i + 1])
            else:
                labels.append(ignore_index)
        pad = seq_length - n
        out["input_ids"].append(toks + [pad_id] * pad)
        out["labels"].append(labels + [ignore_index] * pad)
        out["doc_ids"].append(docs + [-1] * pad)
        out["attention_mask"].append([1] * n + [0] * pad)
    return out

def build_training_data(
    raw_docs: Iterable[str],
    num_merges: int = 100,
    seq_length: int = 128,
    min_confidence: float = 0.7,
) -> Tuple["SimpleTokenizer", Dict[str, List[List[int]]], List[str]]:
    """
    Run the full pipeline and return a trained tokenizer and packed training blocks.

    Order follows the course diagram: clean -> language -> quality -> dedup -> PII -> tokenize -> pack.

    Args:
        raw_docs (Iterable[str]): Raw documents.
        num_merges (int): BPE merges to learn.
        seq_length (int): Block length for packing.
        min_confidence (float): Language confidence threshold.

    Returns:
        Tuple[SimpleTokenizer, Dict[str, List[List[int]]], List[str]]: (tokenizer, packed blocks, final documents).
    """
    filtered = stream_clean(raw_docs, min_confidence=min_confidence)
    deduped = list(stream_dedup(filtered))
    final_docs = [remove_pii(doc) for doc in deduped]
    if not final_docs:
        raise ValueError("no documents survived the pipeline")

    tokenizer = SimpleTokenizer()
    tokenizer.train_bpe(" ".join(final_docs), num_merges=num_merges)
    token_docs = [tokenizer.encode(doc) for doc in final_docs]
    packed = pack_documents(token_docs, tokenizer.eos_id, seq_length, tokenizer.pad_id)
    return tokenizer, packed, final_docs

def compute_statistics(
    documents: List[str],
    token_ids: List[int],
    sequences: List[List[int]],
    tokenizer_vocab_size: int,
    pad_id: int = 0
) -> Dict[str, Any]:
    """
    Computes overall corpus, vocabulary, compression, and sequence padding efficiency metrics.

    Args:
        documents (List[str]): Corpus of cleaned documents.
        token_ids (List[int]): Continuous flat list of token IDs.
        sequences (List[List[int]]): Packed token sequences.
        tokenizer_vocab_size (int): Size of the tokenizer vocabulary.

    Returns:
        Dict[str, Any]: Dictionary containing pre-training dataset analytical statistics.
    """
    total_chars = sum(len(doc) for doc in documents)
    total_bytes = sum(len(doc.encode("utf-8")) for doc in documents)
    total_tokens = len(token_ids)
    unique_tokens = len(set(token_ids))

    slots = sum(len(seq) for seq in sequences)
    non_pad = sum(1 for seq in sequences for token in seq if token != pad_id)

    doc_word_counts = [len(_units(doc)) for doc in documents]

    return {
        "total_documents": len(documents),
        "total_characters": total_chars,
        "total_tokens": total_tokens,
        "unique_tokens": unique_tokens,
        "vocab_utilization": unique_tokens / tokenizer_vocab_size if tokenizer_vocab_size else 0.0,
        "compression_ratio": total_bytes / total_tokens if total_tokens else 0.0,
        "avg_doc_length_words": sum(doc_word_counts) / len(doc_word_counts) if doc_word_counts else 0.0,
        "num_sequences": len(sequences),
        "sequence_utilization": non_pad / slots if slots else 0.0,
        "top_tokens": Counter(token_ids).most_common(10),
    }

# ---------------------------------------------------------------------------
# Doctor-comments corpus (used by mini_gpt_torch.py)
# ---------------------------------------------------------------------------

# Arabic letter variants that appear in user-typed Persian text; NFKC does not unify them.
_TRAILING_PUNCT = " .,\u060c;\u061b:!?\u061f-_~"

def comments_to_documents(
    csv_path: str,
    text_column: str = "text",
    group_column: str = "doctor_id",
    min_words: int = 60,
    max_documents: Optional[int] = None,
    seed: int = 0,
) -> List[str]:
    """
    Turn a CSV of short comments into pre-training documents that the pipeline can accept.

    Doctor comments are tiny (median ~4 words) while quality_filter() requires 50+ words,
    so comments are first de-duplicated (exact match after normalization) and then
    concatenated, per doctor and in file order, into documents of at least `min_words` words.
    A leftover tail shorter than `min_words` is dropped.

    Args:
        csv_path (str): Path to the CSV file.
        text_column (str): Column that holds the comment text.
        group_column (str): Column used to keep comments of the same author together.
        min_words (int): A document is closed as soon as it has this many words.
        max_documents (Optional[int]): If set, keep a random sample of this many documents
            (sampled across all doctors, not just the first ones in the file).
        seed (int): Seed for the sampling/shuffling.

    Returns:
        List[str]: Shuffled list of documents. Comments inside a document are separated by ". ".
    """
    by_group: Dict[str, List[str]] = defaultdict(list)
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            # clean_text() normalizes with hazm before the exact-duplicate check below,
            # so comments that differ only in spelling variants or ZWNJ count as duplicates.
            text = clean_text(row.get(text_column) or "")
            text = text.strip(_TRAILING_PUNCT)
            if text:
                by_group[row.get(group_column, "")].append(text)

    documents: List[str] = []
    for comments in by_group.values():
        seen: Set[str] = set()
        parts: List[str] = []
        word_count = 0
        for comment in comments:
            if comment in seen:
                continue
            seen.add(comment)
            parts.append(comment)
            word_count += len(comment.split())
            if word_count >= min_words:
                documents.append(". ".join(parts) + ".")
                parts, word_count = [], 0

    random.Random(seed).shuffle(documents)
    if max_documents is not None:
        documents = documents[:max_documents]
    return documents

def split_packed(
    packed: Dict[str, List[List[int]]], val_fraction: float = 0.1
) -> Tuple[Dict[str, List[List[int]]], Dict[str, List[List[int]]]]:
    """
    Split packed blocks (output of pack_documents) into a train part and a validation part.

    Documents are already shuffled by comments_to_documents(), so the last blocks are used
    for validation.

    Args:
        packed (Dict[str, List[List[int]]]): Dict with equally long lists per key.
        val_fraction (float): Fraction of blocks reserved for validation (at least one block).

    Returns:
        Tuple[dict, dict]: (train_blocks, val_blocks) with the same keys as `packed`.
    """
    num_blocks = len(next(iter(packed.values())))
    if num_blocks < 2:
        raise ValueError("need at least 2 blocks to split")
    num_val = min(num_blocks - 1, max(1, int(num_blocks * val_fraction)))
    cut = num_blocks - num_val
    train = {k: v[:cut] for k, v in packed.items()}
    val = {k: v[cut:] for k, v in packed.items()}
    return train, val

class PackedBlockLoader(PreTrainingDataLoader):
    """
    PreTrainingDataLoader that also serves labels and doc_ids from pack_documents().

    Shuffling, epochs and rank/world_size sharding are inherited from PreTrainingDataLoader;
    only __iter__ differs: it yields a dict of lists instead of an (inputs, masks) tuple.
    """

    def __init__(
        self,
        packed: Dict[str, List[List[int]]],
        batch_size: int,
        shuffle: bool = True,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
    ):
        super().__init__(packed["input_ids"], packed["attention_mask"], batch_size,
                         shuffle=shuffle, rank=rank, world_size=world_size, seed=seed)
        self.labels = packed["labels"]
        self.doc_ids = packed["doc_ids"]

    def __iter__(self) -> Generator[Dict[str, List[List[int]]], None, None]:
        indices = self._indices()
        for start in range(0, len(indices), self.batch_size):
            batch = indices[start:start + self.batch_size]
            yield {
                "input_ids": [self.sequences[i] for i in batch],
                "labels": [self.labels[i] for i in batch],
                "doc_ids": [self.doc_ids[i] for i in batch],
                "attention_mask": [self.attention_masks[i] for i in batch],
            }