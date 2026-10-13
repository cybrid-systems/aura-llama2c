#!/usr/bin/env python3
"""Generate a tiny llama2.c-format checkpoint with seeded random weights.

Purpose: a seconds-to-load correctness bed for the Aura port (the 260K
stories model takes minutes to load on the Soft interpreter - see
IMPROVEMENTS.md). Same legacy layout as export_weights.py expects,
including the freq_cis_real/imag slots and a SEPARATE classifier
(vocab encoded negative) so the wcls path gets exercised too.

Usage: python3 scripts/make_tiny_model.py <out.bin>
"""
import math
import struct
import sys

# LCG so the run is reproducible without numpy
STATE = 20261013


def rnd():
    global STATE
    STATE = (STATE * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    return ((STATE >> 11) / float(1 << 53)) * 2.0 - 1.0


def main() -> None:
    out = sys.argv[1]
    dim, hidden, n_layers = 16, 32, 2
    n_heads, n_kv_heads, vocab, seq_len = 2, 1, 64, 32
    head_size = dim // n_heads
    kv_dim = head_size * n_kv_heads

    def block(n, scale):
        return [rnd() * scale for _ in range(n)]

    parts = []
    parts.append(block(vocab * dim, 0.05))            # token_embedding_table
    parts.append(block(n_layers * dim, 0.5))          # rms_att
    # wq/wk at scale 1.0 so attention scores are O(1) and softmax is
    # selective - this makes the end-to-end oracle RoPE-sensitive (with the
    # original 0.05 scale every score was ~0, softmax ~uniform, and any
    # RoPE bug was invisible in the logits)
    parts.append(block(n_layers * dim * dim, 1.0))    # wq
    parts.append(block(n_layers * dim * kv_dim, 1.0))  # wk
    parts.append(block(n_layers * dim * kv_dim, 0.05))  # wv
    parts.append(block(n_layers * dim * dim, 0.05))   # wo
    parts.append(block(n_layers * dim, 0.5))          # rms_ffn
    parts.append(block(n_layers * hidden * dim, 0.05))  # w1
    parts.append(block(n_layers * dim * hidden, 0.05))  # w2
    parts.append(block(n_layers * hidden * dim, 0.05))  # w3
    parts.append(block(dim, 0.5))                     # rms_final
    parts.append(block(seq_len * head_size // 2, 1.0))  # freq_cis_real (slot)
    parts.append(block(seq_len * head_size // 2, 1.0))  # freq_cis_imag (slot)
    parts.append(block(vocab * dim, 0.05))            # wcls (separate!)

    with open(out, "wb") as f:
        f.write(struct.pack("7i", dim, hidden, n_layers, n_heads,
                            n_kv_heads, -vocab, seq_len))  # -vocab: unshared
        for part in parts:
            for v in part:
                f.write(struct.pack("<f", v))
    print(f"tiny model -> {out} "
          f"(dim={dim} hidden={hidden} layers={n_layers} "
          f"heads={n_heads}/{n_kv_heads} vocab={vocab} seq={seq_len}, "
          f"wcls separate)")


if __name__ == "__main__":
    main()
