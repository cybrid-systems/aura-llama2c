#!/usr/bin/env python3
"""Export a llama2.c tokenizer.bin to plain text for the Aura port.

Usage: python3 scripts/export_tokenizer.py <tokenizer.bin> <tokenizer.txt>

tokenizer.bin layout (run.c tokenizer_init):
  new format: int magic=20240426, int version, int vocab_size,
              then vocab_size records of [float32 score, int32 len, bytes]
  old format: int magic=0x70726f67, int vocab_size, same records

tokenizer.txt layout:
  line 1: vocab_size
  line 2..: "<score%.9g> <len> <b0> <b1> ..." (bytes as decimal 0..255)
"""
import struct
import sys


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    d = open(src, "rb").read()
    off = 0

    def i32():
        nonlocal off
        (v,) = struct.unpack_from("<i", d, off)
        off += 4
        return v

    def f32():
        nonlocal off
        (v,) = struct.unpack_from("<f", d, off)
        off += 4
        return v

    magic = i32()
    if magic == 20240426:
        version = i32()
        vocab_size = i32()
        print(f"new tokenizer format, version {version}")
    else:
        vocab_size = i32()  # old format: magic 0x70726f67, then vocab_size
        print("old tokenizer format")
    rows = []
    for _ in range(vocab_size):
        score = f32()
        ln = i32()
        piece = d[off:off + ln]
        off += ln
        rows.append((score, piece))
    if off != len(d):
        sys.exit(f"tokenizer layout mismatch: consumed {off} of {len(d)} bytes")
    with open(dst, "w") as f:
        f.write(f"{vocab_size}\n")
        for score, piece in rows:
            f.write("%.9g %d %s\n" % (score, len(piece),
                                      " ".join(str(b) for b in piece)))
    print(f"ok: {src} -> {dst} ({vocab_size} entries)")


if __name__ == "__main__":
    main()
