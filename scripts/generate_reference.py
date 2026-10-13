#!/usr/bin/env python3
"""Full-pipeline generation oracle: encode -> forward -> greedy argmax.

Composes tokenizer_reference.py (same vocab data as the Aura tokenizer) with
reference.py (float64 forward) - the same model file the Aura port loads.

Usage: python3 scripts/generate_reference.py <model.bin> "prompt text" <steps>
Prints the full id sequence (prompt + generated), comma-separated.
"""
import sys

sys.path.insert(0, "scripts")
from tokenizer_reference import load as tok_load, encode  # noqa: E402
from reference import load as model_load, forward  # noqa: E402


def main():
    model_path, text, steps = sys.argv[1], sys.argv[2], int(sys.argv[3])
    vocab_size, scores, pieces, table = tok_load("model/tokenizer.txt")
    m = model_load(model_path)

    ids = encode(text, vocab_size, scores, pieces, table, bos=True)
    pos = 0
    logits = None
    for tid in ids:
        logits = forward(m, tid, pos)
        pos += 1
    # greedy: the logits from the last prompt position give the first
    # generated id; each new id is forwarded exactly once (same as run.c
    # and as lib/l2c/sampler.aura's generate-ids - the earlier version
    # double-forwarded the last prompt token, corrupting the KV cache)
    for _ in range(steps):
        best, bv = 0, logits[0]
        for i in range(1, m.vocab):
            if logits[i] > bv:
                best, bv = i, logits[i]
        ids.append(best)
        logits = forward(m, best, pos)
        pos += 1
    print(",".join(str(i) for i in ids))


if __name__ == "__main__":
    main()
