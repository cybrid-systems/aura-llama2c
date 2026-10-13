#!/usr/bin/env python3
"""Export a llama2.c checkpoint to text weights for the Aura port.

Usage: python3 scripts/export_weights.py <model.bin> <out.weights.txt | --dir <dir>>

Default writes the single-file layout (header + SECTION blocks); `--dir <dir>`
writes per-section files (<dir>/<name>.txt, one float per line, plus
<dir>/config.txt), which is what lib/l2c/model.aura's loader consumes
(per-section files bound live-heap growth during load — see IMPROVEMENTS.md).

The llama2.c binary format (legacy layout, which is what the tinyllamas
checkpoints use):

  28-byte header: 7 little-endian int32
      dim hidden_dim n_layers n_heads n_kv_heads vocab_size seq_len
      (vocab_size is NEGATIVE when the classifier weights are stored
       separately; positive means wcls == token embedding table)

  then float32 weights, sequentially:
      token_embedding_table  (vocab*dim)
      rms_att_weight         (layer*dim)
      wq                     (layer*dim*(n_heads*head_size))
      wk                     (layer*dim*(n_kv_heads*head_size))
      wv                     (layer*dim*(n_kv_heads*head_size))
      wo                     (layer*(n_heads*head_size)*dim)
      rms_ffn_weight         (layer*dim)
      w1                     (layer*hidden_dim*dim)
      w2                     (layer*dim*hidden_dim)
      w3                     (layer*hidden_dim*dim)
      rms_final_weight       (dim)
      freq_cis_real          (seq_len*head_size/2)   [legacy, unused: RoPE is
      freq_cis_imag          (seq_len*head_size/2)    computed on the fly]
      wcls                   (vocab*dim, only when unshared)

Floats are printed with %.9g (exact float32 round-trip; the Aura runtime
parses them as doubles). freq_cis_real/imag are skipped and wcls is skipped
when shared.
"""
import os
import struct
import sys


def main() -> None:
    if len(sys.argv) not in (3, 4):
        sys.exit("usage: export_weights.py <model.bin> <out.weights.txt | --dir <dir>>")
    src = sys.argv[1]
    out_dir = None
    if len(sys.argv) == 4 and sys.argv[2] == "--dir":
        out_dir = sys.argv[3]
        os.makedirs(out_dir, exist_ok=True)
        dst = None
    else:
        dst = sys.argv[2]

    data = open(src, "rb").read()
    dim, hidden, n_layers, n_heads, n_kv_heads, vocab, seq_len = struct.unpack(
        "7i", data[:28])
    shared = vocab > 0
    vocab = abs(vocab)
    head_size = dim // n_heads
    kv_dim = head_size * n_kv_heads
    floats = struct.unpack(f"<{(len(data) - 28) // 4}f", data[28:])

    pos = 0
    out = None if out_dir else open(dst, "w")

    def write_values(chunk, name):
        if out_dir:
            with open(os.path.join(out_dir, name + ".txt"), "w") as f:
                for v in chunk:
                    f.write("%.9g\n" % v)
        else:
            out.write(f"SECTION {name} {len(chunk)}\n")
            for v in chunk:
                out.write("%.9g\n" % v)

    def take(name, count, skip=False):
        nonlocal pos
        chunk = floats[pos:pos + count]
        if len(chunk) != count:
            sys.exit(f"truncated checkpoint at section {name}")
        if not skip:
            write_values(chunk, name)
        pos += count

    header = (f"{dim} {hidden} {n_layers} {n_heads} {n_kv_heads} "
              f"{vocab} {seq_len}")
    if out_dir:
        with open(os.path.join(out_dir, "config.txt"), "w") as cfg:
            cfg.write(header + "\n")
    else:
        out.write(header + "\n")

    take("tok_emb", vocab * dim)
    take("rms_att", n_layers * dim)
    take("wq", n_layers * dim * dim)
    take("wk", n_layers * dim * kv_dim)
    take("wv", n_layers * dim * kv_dim)
    take("wo", n_layers * dim * dim)
    take("rms_ffn", n_layers * dim)
    take("w1", n_layers * hidden * dim)
    take("w2", n_layers * dim * hidden)
    take("w3", n_layers * hidden * dim)
    take("rms_final", dim)
    take("freq_cis_real", seq_len * head_size // 2, skip=True)
    take("freq_cis_imag", seq_len * head_size // 2, skip=True)
    if not shared:
        take("wcls", vocab * dim)
    if out:
        out.close()

    if pos != len(floats):
        sys.exit(f"layout mismatch: consumed {pos} of {len(floats)} floats")
    print(f"ok: {src} -> {out_dir or dst} ({len(floats)} floats, "
          f"dim={dim} hidden={hidden} layers={n_layers} "
          f"heads={n_heads}/{n_kv_heads} vocab={vocab} seq={seq_len}, "
          f"wcls {'shared' if shared else 'separate'})")


if __name__ == "__main__":
    main()
