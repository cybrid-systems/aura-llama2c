#!/usr/bin/env python3
"""Python oracle for the Aura BPE tokenizer (run.c encode/decode semantics).

Consumes model/tokenizer.txt (same data as lib/l2c/tokenizer.aura).

encode: [BOS?] + [dummy " " prefix if text non-empty] + per-code-point vocab
lookup (byte fallback: id = byte+3) + pair-merge loop merging the
consecutive pair with the highest score until none merges.

decode: ids <3 skipped; ids 3..258 are byte-fallback pieces (byte = id-3);
other ids emit their piece bytes, stripping a leading space right after BOS.

Usage:
  python3 scripts/tokenizer_reference.py encode "Once upon a time"   -> ids
  python3 scripts/tokenizer_reference.py decode 1,59,420            -> text
"""
import sys


def load(path="model/tokenizer.txt"):
    lines = open(path).read().splitlines()
    n = int(lines[0])
    scores = [0.0] * n
    pieces = [""] * n
    for i in range(n):
        parts = lines[1 + i].split(" ")
        scores[i] = float(parts[0])
        ln = int(parts[1])
        bs = [int(x) for x in parts[2:2 + ln]]
        pieces[i] = bytes(bs).decode("latin-1")  # code points == byte values
    table = {p: i for i, p in enumerate(pieces)}
    return n, scores, pieces, table


def encode(text, vocab_size, scores, pieces, table, bos=True):
    ids = []
    if bos:
        ids.append(1)
    if text:
        if " " in table:
            ids.append(table[" "])
    for ch in text:
        bls = list(ch.encode("utf-8"))
        key = bytes(bls).decode("latin-1")
        if key in table:
            ids.append(table[key])
        else:
            ids.extend(b + 3 for b in bls)
    # pair-merge loop
    while True:
        best_score = -1e10
        best_id = -1
        best_idx = -1
        for i in range(len(ids) - 1):
            cat = pieces[ids[i]] + pieces[ids[i + 1]]
            j = table.get(cat, -1)
            if j != -1 and scores[j] > best_score:
                best_score = scores[j]
                best_id = j
                best_idx = i
        if best_idx == -1:
            break
        ids[best_idx] = best_id
        del ids[best_idx + 1]
    return ids


def decode(ids, pieces):
    out = []
    prev = -1
    for tid in ids:
        if tid < 3:
            prev = tid
            continue
        if tid <= 258:
            out.append(chr(tid - 3))
        else:
            piece = pieces[tid]
            if prev == 1 and piece.startswith(" "):
                piece = piece[1:]
            out.append(piece)
        prev = tid
    return "".join(out)


def main():
    n, scores, pieces, table = load()
    mode = sys.argv[1]
    if mode == "encode":
        bos = sys.argv[3] != "0" if len(sys.argv) > 3 else True
        print(",".join(str(i) for i in encode(sys.argv[2], n, scores,
                                              pieces, table, bos)))
    elif mode == "decode":
        ids = [int(x) for x in sys.argv[2].split(",")]
        sys.stdout.write(decode(ids, pieces))
        sys.stdout.write("\n")
    else:
        sys.exit("unknown mode")


if __name__ == "__main__":
    main()
