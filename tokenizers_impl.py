"""
tokenizers_impl.py
==================
Three subword tokenization strategies built on a common abstract base class.

Background — why subword tokenization?
---------------------------------------
Early NLP systems split text on whitespace (word-level) or on individual
characters (character-level).  Both extremes have problems:

  * Word-level: vocabulary explodes for morphologically rich languages;
    out-of-vocabulary (OOV) words get a single <UNK> token, losing all
    information.
  * Character-level: sequences become very long; the model must learn
    word structure from scratch.

Subword tokenization is the middle ground: common words stay as single
tokens, rare or unknown words are split into smaller, reusable pieces.
All three algorithms below operate in this space but differ in *how* they
learn the vocabulary and *how* they split at inference time.

Algorithms implemented
-----------------------
  1. BPETokenizer      — Byte-Pair Encoding (Sennrich et al., 2016)
  2. WordPieceTokenizer — WordPiece (Schuster & Nakamura, 2012; used in BERT)
  3. SentencePieceTokenizer — SentencePiece (Kudo & Richardson, 2018)

Each class wraps the battle-tested HuggingFace `tokenizers` library (BPE,
WordPiece) or Google's `sentencepiece` library, so the heavy lifting is
done by optimised C++ backends.
"""

from __future__ import annotations

import abc
import io
import os
import tempfile
from typing import List

# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

_MODEL_DIR = ".model"


class BaseTokenizer(abc.ABC):

    @staticmethod
    def _ensure_model_dir() -> str:
        """Create ``.model/`` in the cwd if needed and return its path."""
        os.makedirs(_MODEL_DIR, exist_ok=True)
        return _MODEL_DIR


    @abc.abstractmethod
    def train(self, texts: List[str], vocab_size: int = 1000) -> None:
        """
        Learn a subword vocabulary from *texts*.

        Parameters
        ----------
        texts : list[str]
            Raw training corpus.  More text → better coverage.
        vocab_size : int
            Target number of distinct tokens in the learned vocabulary.
            Larger values reduce splitting of rare words but increase
            memory and model size.
        """

    @abc.abstractmethod
    def encode(self, text: str) -> List[str]:
        """
        Tokenize *text* using the learned vocabulary.

        Parameters
        ----------
        text : str
            Any UTF-8 string.

        Returns
        -------
        list[str]
            Ordered list of subword token strings.
        """

    @abc.abstractmethod
    def decode(self, tokens: List[str]) -> str:
        """
        Reconstruct a string from a list of subword tokens.

        Parameters
        ----------
        tokens : list[str]
            Output of a previous encode() call (or compatible tokens).

        Returns
        -------
        str
            Reconstructed text.  May differ from the original in
            whitespace normalisation depending on the algorithm.
        """

    @abc.abstractmethod
    def save(self, path: str) -> None:
        """Persist the trained model to *path* (a file or directory)."""

    @abc.abstractmethod
    def load(self, path: str) -> None:
        """Restore a previously saved model from *path*."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ---------------------------------------------------------------------------
# 1. Byte-Pair Encoding
# ---------------------------------------------------------------------------

class BPETokenizer(BaseTokenizer):
    """
    Byte-Pair Encoding (BPE) tokenizer.

    Algorithm overview
    ------------------
    BPE was originally a data-compression algorithm (Philip Gage, 1994)
    adapted for NLP by Sennrich et al. (2016) for neural machine translation.

    Training (offline, done once):
      1. Start with a character-level vocabulary: every unique character in
         the corpus becomes a token, plus a special end-of-word marker
         (commonly "Ġ" or "</w>").
      2. Count every adjacent pair of tokens across the entire corpus.
      3. Merge the most frequent pair into a single new token.
      4. Repeat steps 2–3 until the vocabulary reaches `vocab_size`.

    The result is a *merge table*: an ordered list of (A, B) → AB rules.

    Encoding (inference):
      Apply the merge rules in the same order they were learned.  A word
      that was common in training stays as one token; a rare word gets split
      into its constituent learned pieces.

    Key properties
    --------------
    * Deterministic: the same string always produces the same tokens.
    * Greedy: merges are applied left-to-right in priority order.
    * Language-agnostic: works on raw bytes (GPT-2 uses byte-level BPE),
      so it handles any Unicode text without a pre-tokenisation step.
    * Used by: GPT-2, GPT-3, GPT-4, RoBERTa, BART, LLaMA.

    Complexity
    ----------
    Training: O(V · N) where V = merge steps, N = corpus size.
    Encoding: O(T · M) where T = tokens in string, M = merge table size.

    Implementation note
    -------------------
    This class wraps `tokenizers.models.BPE` from HuggingFace, which
    implements the algorithm in Rust for speed.  We use a
    `ByteLevelBPETokenizer` trainer so the tokenizer operates on UTF-8
    bytes, making it robust to any language or emoji.
    """

    _DEFAULT_SAVE_PATH = "bpe_tokenizer.json"

    def __init__(self) -> None:
        self._tokenizer = None  # set after train()

    def train(self, texts: List[str], vocab_size: int = 1000) -> None:
        """
        Learn BPE merge rules from *texts*.

        Internally writes the corpus to a temporary file because the
        HuggingFace trainer expects file paths.  Saves the model to
        ``_DEFAULT_SAVE_PATH`` on success.
        """
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder

        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = ByteLevel()
        tokenizer.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["[UNK]"],
            show_progress=False,
        )

        # Write corpus to a temp file; trainer reads line-by-line.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                         delete=False, encoding="utf-8") as f:
            f.write("\n".join(texts))
            tmp_path = f.name

        try:
            tokenizer.train([tmp_path], trainer)
        finally:
            os.unlink(tmp_path)

        self._tokenizer = tokenizer
        self.save(os.path.join(self._ensure_model_dir(), self._DEFAULT_SAVE_PATH))

    def save(self, path: str) -> None:
        """Save the tokenizer to a JSON file at *path*."""
        if self._tokenizer is None:
            raise RuntimeError("Nothing to save — call train() first.")
        self._tokenizer.save(path)

    def load(self, path: str) -> None:
        """Load a previously saved tokenizer from a JSON file at *path*."""
        from tokenizers import Tokenizer
        self._tokenizer = Tokenizer.from_file(path)

    def encode(self, text: str) -> List[str]:
        """Apply learned BPE merges to *text* and return token strings."""
        if self._tokenizer is None:
            raise RuntimeError("Call train() before encode().")
        return self._tokenizer.encode(text).tokens

    def decode(self, tokens: List[str]) -> str:
        """Reconstruct text from BPE tokens using the byte-level decoder."""
        if self._tokenizer is None:
            raise RuntimeError("Call train() before decode().")
        ids = [self._tokenizer.token_to_id(t) for t in tokens]
        return self._tokenizer.decode(ids)


# ---------------------------------------------------------------------------
# 2. WordPiece
# ---------------------------------------------------------------------------

class WordPieceTokenizer(BaseTokenizer):
    """
    WordPiece tokenizer.

    Algorithm overview
    ------------------
    WordPiece was developed at Google for Japanese/Korean segmentation
    (Schuster & Nakamura, 2012) and later popularised by BERT
    (Devlin et al., 2018).

    Training:
      Like BPE, WordPiece starts from characters and iteratively merges
      pairs.  The key difference is the *merge criterion*:

        BPE picks the pair with the highest raw frequency.
        WordPiece picks the pair that maximises the *likelihood* of the
        training corpus under a unigram language model:

            score(A, B) = freq(AB) / (freq(A) × freq(B))

      This ratio is high when A and B co-occur more than chance predicts,
      so WordPiece prefers merges that carry the most mutual information.

    Encoding (inference):
      WordPiece uses a *longest-match-first* (greedy) strategy per word:
        1. Try to match the longest prefix of the remaining string that
           exists in the vocabulary.
        2. If found, emit that token; advance past it.
        3. If no prefix matches, emit [UNK].
      Continuation sub-tokens are prefixed with "##" to signal that they
      are not word-initial (e.g., "playing" → ["play", "##ing"]).

    Key properties
    --------------
    * The "##" prefix encodes position-within-word information, which
      helps models distinguish "un##" (suffix) from "un" (prefix).
    * Requires whitespace pre-tokenisation: words are split on spaces
      first, then each word is sub-tokenised independently.
    * Used by: BERT, DistilBERT, ELECTRA, MobileBERT.

    Complexity
    ----------
    Training: O(V · N) — same asymptotic as BPE but with a more expensive
    per-step score computation.
    Encoding: O(W · L²) where W = words, L = max word length (due to
    longest-match scan).
    """

    _DEFAULT_SAVE_PATH = "wordpiece_tokenizer.json"

    def __init__(self) -> None:
        self._tokenizer = None

    def train(self, texts: List[str], vocab_size: int = 1000) -> None:
        """
        Learn a WordPiece vocabulary from *texts*.

        Uses HuggingFace's `WordPieceTrainer`, which implements the
        likelihood-maximisation criterion described above.  Saves the model
        to ``_DEFAULT_SAVE_PATH`` on success.
        """
        from tokenizers import Tokenizer
        from tokenizers.models import WordPiece
        from tokenizers.trainers import WordPieceTrainer
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.decoders import WordPiece as WordPieceDecoder

        tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.decoder = WordPieceDecoder()

        trainer = WordPieceTrainer(
            vocab_size=vocab_size,
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
            show_progress=False,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                         delete=False, encoding="utf-8") as f:
            f.write("\n".join(texts))
            tmp_path = f.name

        try:
            tokenizer.train([tmp_path], trainer)
        finally:
            os.unlink(tmp_path)

        self._tokenizer = tokenizer
        self.save(os.path.join(self._ensure_model_dir(), self._DEFAULT_SAVE_PATH))

    def save(self, path: str) -> None:
        """Save the tokenizer to a JSON file at *path*."""
        if self._tokenizer is None:
            raise RuntimeError("Nothing to save — call train() first.")
        self._tokenizer.save(path)

    def load(self, path: str) -> None:
        """Load a previously saved tokenizer from a JSON file at *path*."""
        from tokenizers import Tokenizer
        self._tokenizer = Tokenizer.from_file(path)

    def encode(self, text: str) -> List[str]:
        """
        Tokenize *text* with longest-match-first WordPiece decoding.
        Continuation pieces are prefixed with '##'.
        """
        if self._tokenizer is None:
            raise RuntimeError("Call train() before encode().")
        return self._tokenizer.encode(text).tokens

    def decode(self, tokens: List[str]) -> str:
        """
        Reconstruct text by stripping '##' prefixes and joining with spaces
        where appropriate.
        """
        if self._tokenizer is None:
            raise RuntimeError("Call train() before decode().")
        ids = [self._tokenizer.token_to_id(t) for t in tokens]
        return self._tokenizer.decode(ids)


# ---------------------------------------------------------------------------
# 3. SentencePiece
# ---------------------------------------------------------------------------

class SentencePieceTokenizer(BaseTokenizer):
    """
    SentencePiece tokenizer (Unigram Language Model variant).

    Algorithm overview
    ------------------
    SentencePiece (Kudo & Richardson, 2018) differs from BPE and WordPiece
    in two fundamental ways:

    1. No pre-tokenisation on whitespace.
       BPE and WordPiece first split on spaces, then sub-tokenise each word.
       SentencePiece treats the raw byte stream (including spaces) as input,
       encoding spaces as a special "▁" (U+2581) character.  This makes it
       truly language-agnostic — it works equally well on Chinese, Japanese,
       Thai, or any language that does not use spaces as word boundaries.

    2. Unigram Language Model (ULM) training.
       Instead of iteratively *merging* pairs (bottom-up like BPE), ULM
       works top-down:
         a. Start with a large seed vocabulary (all substrings up to a
            length limit).
         b. Assign each token a log-probability under a unigram model.
         c. Compute the *loss* = negative log-likelihood of the corpus
            given the current vocabulary.
         d. Remove the tokens whose removal increases loss the least
            (i.e., the least useful tokens).
         e. Repeat until the vocabulary reaches `vocab_size`.

       At inference, the tokenisation that *maximises the probability* of
       the input string under the unigram model is chosen via the Viterbi
       algorithm — this is the key difference from BPE's greedy merges.

    Key properties
    --------------
    * Probabilistic: multiple segmentations are possible; the most likely
      one is returned.  This can be used for data augmentation by sampling
      from the distribution (SentencePiece supports this natively).
    * Space-aware: "▁" prefix on tokens marks word boundaries without
      requiring a separate pre-tokeniser.
    * Reversible: encode → decode is lossless (spaces are preserved via ▁).
    * Used by: T5, ALBERT, XLNet, mBART, LLaMA (with BPE variant), Gemma.

    Complexity
    ----------
    Training: O(V² · N / V) ≈ O(V · N) — EM-style iterations over corpus.
    Encoding: O(L²) per sentence via Viterbi DP where L = sentence length
    in characters.

    Implementation note
    -------------------
    Google's `sentencepiece` Python package wraps a C++ library.  Training
    requires writing the corpus to disk; the model is serialised to a
    `.model` file which is then loaded for encoding/decoding.
    """

    _DEFAULT_SAVE_PATH = "sentencepiece_tokenizer.model"

    def __init__(self) -> None:
        self._sp = None
        self._model_path = None

    def train(self, texts: List[str], vocab_size: int = 1000) -> None:
        """
        Train a Unigram SentencePiece model on *texts*.

        The model file is written to a temporary directory and kept in
        memory for subsequent encode/decode calls.  Saves the model to
        ``_DEFAULT_SAVE_PATH`` on success.

        We use the `sentencepiece` library instead of the `tokenizers` library because:
        - tokenizers implements BPE and WordPiece natively in Rust. It does have a tokenizers.models.Unigram model, but its SentencePiece support is primarily for loading pre-trained .model
        files, not training from scratch with the full SentencePiece trainer spec.
        - sentencepiece (Google) is the canonical implementation — it supports training the Unigram LM from scratch with all the original options (character_coverage, byte_fallback,
        split_by_unicode_script, etc.).
        """
        import sentencepiece as spm

        # SentencePiece trainer reads from a file.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                         delete=False, encoding="utf-8") as f:
            f.write("\n".join(texts))
            corpus_path = f.name

        model_prefix = corpus_path.replace(".txt", "_sp")

        spm.SentencePieceTrainer.train(
            input=corpus_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="unigram",   # ULM variant
            character_coverage=0.9995,
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        )

        os.unlink(corpus_path)

        self._model_path = model_prefix + ".model"
        self._sp = spm.SentencePieceProcessor()
        self._sp.load(self._model_path)
        self.save(os.path.join(self._ensure_model_dir(), self._DEFAULT_SAVE_PATH))

    def save(self, path: str) -> None:
        """Copy the trained ``.model`` file to *path*."""
        import shutil
        if self._model_path is None or not os.path.exists(self._model_path):
            raise RuntimeError("Nothing to save — call train() first.")
        shutil.copy2(self._model_path, path)

    def load(self, path: str) -> None:
        """Load a SentencePiece model from *path*."""
        import sentencepiece as spm
        self._sp = spm.SentencePieceProcessor()
        self._sp.load(path)
        self._model_path = path

    def encode(self, text: str) -> List[str]:
        """
        Tokenize *text* using Viterbi-optimal Unigram segmentation.
        Spaces are represented as leading '▁' on tokens.
        """
        if self._sp is None:
            raise RuntimeError("Call train() before encode().")
        return self._sp.encode(text, out_type=str)

    def decode(self, tokens: List[str]) -> str:
        """
        Reconstruct text from SentencePiece tokens.
        '▁' markers are converted back to spaces.
        """
        if self._sp is None:
            raise RuntimeError("Call train() before decode().")
        return self._sp.decode(tokens)

    def __del__(self) -> None:
        # Only clean up temp model files (not user-saved ones inside .model/).
        saved_path = os.path.join(_MODEL_DIR, self._DEFAULT_SAVE_PATH)
        if (self._model_path and os.path.exists(self._model_path)
                and self._model_path != saved_path):
            os.unlink(self._model_path)
        vocab_path = (self._model_path or "").replace(".model", ".vocab")
        saved_vocab = saved_path.replace(".model", ".vocab")
        if os.path.exists(vocab_path) and vocab_path != saved_vocab:
            os.unlink(vocab_path)
