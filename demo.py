"""
demo.py
=======
Fetches a passage of text from Project Gutenberg (public domain) and runs
all three tokenizers on it, printing a side-by-side comparison.

What kinds of text can you use?
--------------------------------
These tokenizers are general-purpose subword models.  They work well on:

  GOOD FITS
  ---------
  * Natural language prose (novels, news articles, Wikipedia) — the primary
    use case.  The more text you train on, the better the vocabulary covers
    common words as single tokens.

  * Multilingual corpora — SentencePiece in particular handles scripts that
    don't use spaces (Chinese, Japanese, Thai) because it treats the raw
    character stream as input.  BPE with byte-level encoding also handles
    any Unicode text.

  * Code — works, but dedicated tokenizers (e.g., CodeBPE) trained on code
    corpora will produce cleaner splits along identifier boundaries.

  * Social media / informal text — handles emoji and slang via byte-level
    BPE; WordPiece may produce many [UNK] tokens if the training corpus
    was formal text.

  POOR FITS (without adaptation)
  --------------------------------
  * Highly structured data (CSV, JSON, XML) — tokenizers will split field
    names and punctuation in unintuitive ways.  Use a parser instead.

  * Binary / non-UTF-8 data — SentencePiece and WordPiece expect valid
    UTF-8.  Byte-level BPE can handle arbitrary bytes but the tokens won't
    be meaningful.

  * Very short texts (< ~1 000 words for training) — the vocabulary will
    overfit to the training sample and produce poor splits on new text.
    Use a pre-trained tokenizer (e.g., from HuggingFace Hub) instead.

  VOCAB SIZE GUIDANCE
  --------------------
  * 500–2 000  : toy / demo (used here)
  * 8 000–32 000 : typical single-language model (BERT uses 30 522)
  * 50 000–100 000 : multilingual or large LLMs (GPT-4 uses ~100 256)
"""

import textwrap
import requests
from tokenizers_impl import BPETokenizer, WordPieceTokenizer, SentencePieceTokenizer

# ---------------------------------------------------------------------------
# 1. Fetch training + demo text from Project Gutenberg
#    "Alice's Adventures in Wonderland" — short, public domain, plain text.
# ---------------------------------------------------------------------------
GUTENBERG_URL = "https://www.gutenberg.org/files/11/11-0.txt"

print("Fetching text from Project Gutenberg...")
response = requests.get(GUTENBERG_URL, timeout=15)
response.encoding = "utf-8"
full_text = response.text

# Strip the Gutenberg header/footer boilerplate (between *** START *** markers).
start = full_text.find("*** START OF THE PROJECT GUTENBERG EBOOK")
end   = full_text.find("*** END OF THE PROJECT GUTENBERG EBOOK")
book_text = full_text[start:end] if start != -1 else full_text

# Split into non-empty lines for training.
lines = [l.strip() for l in book_text.splitlines() if l.strip()]
print(f"  Loaded {len(lines):,} lines ({len(book_text):,} characters)\n")

# ---------------------------------------------------------------------------
# 2. Train all three tokenizers on the full book text.
# ---------------------------------------------------------------------------
VOCAB_SIZE = 1000   # small for demo speed; real models use 8k–100k

tokenizers = {
    "BPE":          BPETokenizer(),
    "WordPiece":    WordPieceTokenizer(),
    "SentencePiece": SentencePieceTokenizer(),
}

for name, tok in tokenizers.items():
    print(f"Training {name}...")
    tok.train(lines, vocab_size=VOCAB_SIZE)
print()

# ---------------------------------------------------------------------------
# 3. Encode a few sample sentences and print results.
# ---------------------------------------------------------------------------
samples = [
    "Alice was beginning to get very tired of sitting by her sister on the bank.",
    "Curiosity killed the cat, but satisfaction brought it back.",
    "The tokenization of subwords is fundamental to modern NLP pipelines.",
    "Supercalifragilisticexpialidocious is a very long and unusual word.",
]

SEP = "─" * 72

for sentence in samples:
    print(SEP)
    print(f"INPUT : {sentence}\n")
    for name, tok in tokenizers.items():
        tokens  = tok.encode(sentence)
        decoded = tok.decode(tokens)
        print(f"  {name:<14} ({len(tokens):>2} tokens)")
        # Wrap long token lists for readability
        token_str = " | ".join(tokens)
        for line in textwrap.wrap(token_str, width=68, subsequent_indent=" " * 18):
            print(f"    tokens : {line}")
        print(f"    decoded: {decoded}")
        print()

print(SEP)
print("Done.")
