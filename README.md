# TokenizationMethods

Implementations of three subword tokenization algorithms — BPE, WordPiece, and SentencePiece — built on a common abstract base class.

## Algorithms

| Algorithm                  | Library backend            | Used by                   |
| -------------------------- | -------------------------- | ------------------------- |
| Byte-Pair Encoding (BPE)   | HuggingFace `tokenizers` | GPT-2/3/4, LLaMA, RoBERTa |
| WordPiece                  | HuggingFace `tokenizers` | BERT, DistilBERT, ELECTRA |
| SentencePiece (Unigram LM) | Google `sentencepiece`   | T5, ALBERT, mBART, Gemma  |

## Setup

```bash
python -m venv .venv # or python3 -m venv .venv
source .venv/bin/activate
pip install sentencepiece tokenizers requests
```

## Usage

### As a library

```python
from tokenizers_impl import BPETokenizer, WordPieceTokenizer, SentencePieceTokenizer

corpus = ["Alice was beginning to get very tired of sitting by her sister on the bank."]

tok = BPETokenizer()
tok.train(corpus, vocab_size=1000)

tokens = tok.encode("Alice was beginning to get very tired")
# ['ĠAlice', 'Ġwas', 'Ġbeginning', 'Ġto', 'Ġget', 'Ġvery', 'Ġti', 'red']

text = tok.decode(tokens)
# 'Alice was beginning to get very tired'
```

All three classes share the same interface — swap `BPETokenizer` for `WordPieceTokenizer` or `SentencePieceTokenizer` without changing any other code.

### Demo script

Fetches *Alice in Wonderland* from Project Gutenberg, trains all three tokenizers, and prints a side-by-side comparison:

```bash
python demo.py
```

### CLI

```bash
# Encode with BPE (defaults: --tokenizer BPE --mode encode)
python tokenize_cli.py --text "Hello world"

# Encode with WordPiece
python tokenize_cli.py --tokenizer WordPiece --text "Hello world"

# Encode with SentencePiece
python tokenize_cli.py --tokenizer SentencePiece --text "Hello world"

# Decode (pass space-separated tokens)
python tokenize_cli.py --tokenizer BPE --mode decode --text "ĠH ell o Ġwor ld"
python tokenize_cli.py --tokenizer WordPiece --mode decode --text "He ##ll ##o wor ##ld"
python tokenize_cli.py --tokenizer SentencePiece --mode decode --text "▁He ll o ▁world"
```

#### CLI flags

| Flag            | Required      | Default    | Description                                                 |
| --------------- | ------------- | ---------- | ----------------------------------------------------------- |
| `--tokenizer` | No            | `BPE`    | `BPE`, `WordPiece`, or `SentencePiece`                |
| `--text`      | **Yes** | —         | Input string to encode, or space-separated tokens to decode |
| `--mode`      | No            | `encode` | `encode` or `decode`                                    |

> **Note:** the CLI trains on *Alice in Wonderland* on every run (~5s). This keeps the tool self-contained without requiring a pre-saved model file.

## Project structure

```
TokenizationMethods/
├── tokenizers_impl.py   # BaseTokenizer + BPE, WordPiece, SentencePiece classes
├── demo.py              # End-to-end demo on Project Gutenberg text
├── tokenize_cli.py      # Command-line interface
└── README.md
```
