#!/usr/bin/env python3
"""Extract the sentencepiece vocab (tok512.model) into:
  1. model/tokenizer.txt  - the Aura port's text format:
       line 1: vocab_size
       per entry: "<score%.9g> <len> <b0> <b1> ..." (bytes decimal 0..255)
     Byte-fallback pieces (<0xXX>) are stored in run.c's 2-byte form
     [0xC0|XX, 0x80|XX]; decode skips ids < 3 (special tokens).
  2. model/tokenizer_llama2c.bin - same vocab in the format compiled run.c
     (karpathy/llama2.c master) reads: int max_token_length, then records
     of [score f32][len i32][bytes]; the vocab size comes from the model
     config, so run.c can load this directly as its -z tokenizer.

Usage: python3 scripts/export_tokenizer_sp.py <tok512.model>
"""
import struct
import sys

import sentencepiece as spm


def main() -> None:
    model_path = sys.argv[1]
    sp = spm.SentencePieceProcessor(model_file=model_path)
    n = sp.get_piece_size()
    assert n == 512, n

    entries = []
    for i in range(n):
        piece = sp.id_to_piece(i)
        score = sp.get_score(i)
        # llama2.c tokenizer convention: sentencepiece's U+2581 marker becomes
        # a literal space, so run.c's dummy_prefix str_lookup(" ") finds it
        # (without this, encode emits -1 and run.c segfaults on vocab[-1])
        piece = piece.replace("\u2581", " ")
        if len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">"):
            b = int(piece[3:5], 16)
            raw = bytes([0xC0 | b, 0x80 | b])
        else:
            raw = piece.encode("utf-8")
        entries.append((score, raw))

    with open("model/tokenizer.txt", "w") as f:
        f.write(f"{n}\n")
        for score, raw in entries:
            f.write("%.9g %d %s\n" % (score, len(raw),
                                      " ".join(str(b) for b in raw)))

    max_len = max(len(raw) for _, raw in entries)
    with open("model/tokenizer_llama2c.bin", "wb") as f:
        f.write(struct.pack("<i", max_len))
        for score, raw in entries:
            f.write(struct.pack("<fi", score, len(raw)))
            f.write(raw)
    print(f"ok: {model_path} -> model/tokenizer.txt + model/tokenizer_llama2c.bin "
          f"({n} entries)")


if __name__ == "__main__":
    main()
