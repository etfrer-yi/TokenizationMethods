"""
tokenize_cli.py
---------------
CLI for encoding or decoding text with BPE, WordPiece, or SentencePiece.

The tokenizer is trained on-the-fly using Alice in Wonderland (Project
Gutenberg) as the vocabulary corpus before the requested operation is applied.

Usage examples
--------------
  # Encode with BPE (default tokenizer and mode)
  python tokenize_cli.py --tokenizer BPE --text "Hello world"

  # Encode with WordPiece
  python tokenize_cli.py --tokenizer WordPiece --text "Hello world" --mode encode

  # Decode with SentencePiece (tokens separated by spaces)
  python tokenize_cli.py --tokenizer SentencePiece --text "▁Hello ▁world" --mode decode
"""

import argparse
import requests
from tokenizers_impl import BPETokenizer, WordPieceTokenizer, SentencePieceTokenizer

TOKENIZER_MAP = {
    "BPE": BPETokenizer,
    "WordPiece": WordPieceTokenizer,
    "SentencePiece": SentencePieceTokenizer,
}

GUTENBERG_URL = "https://www.gutenberg.org/files/11/11-0.txt"
VOCAB_SIZE = 1000


def fetch_training_corpus() -> list[str]:
    response = requests.get(GUTENBERG_URL, timeout=15)
    response.encoding = "utf-8"
    text = response.text
    start = text.find("*** START OF THE PROJECT GUTENBERG EBOOK")
    end   = text.find("*** END OF THE PROJECT GUTENBERG EBOOK")
    book  = text[start:end] if start != -1 else text
    return [l.strip() for l in book.splitlines() if l.strip()]


def main():
    parser = argparse.ArgumentParser(description="Subword tokenizer CLI")
    parser.add_argument(
        "--tokenizer", choices=TOKENIZER_MAP.keys(), default="BPE",
        help="Tokenization algorithm to use (default: BPE)",
    )
    parser.add_argument(
        "--text", required=True,
        help="Text to encode, or space-separated tokens to decode",
    )
    parser.add_argument(
        "--mode", choices=["encode", "decode"], default="encode",
        help="Operation mode: encode (default) or decode",
    )
    args = parser.parse_args()

    print(f"Fetching training corpus...", flush=True)
    corpus = fetch_training_corpus()

    tok = TOKENIZER_MAP[args.tokenizer]()
    print(f"Training {args.tokenizer} tokenizer on {len(corpus):,} lines...", flush=True)
    tok.train(corpus, vocab_size=VOCAB_SIZE)

    if args.mode == "encode":
        tokens = tok.encode(args.text)
        print(f"\nTokens ({len(tokens)}): {tokens}")
    else:
        # Decode expects a list of token strings; split the input on spaces.
        tokens = args.text.split(" ")
        result = tok.decode(tokens)
        print(f"\nDecoded: {result}")


if __name__ == "__main__":
    main()
