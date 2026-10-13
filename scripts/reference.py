#!/usr/bin/env python3
"""Pure-python float64 reference forward for llama2.c checkpoints.

Mirrors run.c's forward() (RoPE on the fly, GQA, silu FFN) but accumulates
in float64 and reads the legacy binary layout (see export_weights.py).
Prints, per position: "pos <i> argmax <id>" followed by vocab_size lines of
logits (%.17g) - byte-compatible layout with the Aura driver output.

Usage: python3 scripts/reference.py <model.bin> <token_ids_commas> [extra_steps]
"""
import math
import struct
import sys


class M:
    pass


def load(path):
    m = M()
    d = open(path, "rb").read()
    (m.dim, m.hidden, m.n_layers, m.n_heads, m.n_kv_heads, m.vocab,
     m.seq_len) = struct.unpack("7i", d[:28])
    m.shared = m.vocab > 0
    m.vocab = abs(m.vocab)
    m.head_size = m.dim // m.n_heads
    m.kv_dim = m.head_size * m.n_kv_heads
    f = struct.unpack(f"<{(len(d) - 28) // 4}f", d[28:])
    p = 0

    def take(n):
        nonlocal p
        c = f[p:p + n]
        p += n
        return c

    m.emb = take(m.vocab * m.dim)
    m.rms_att = take(m.n_layers * m.dim)
    m.wq = take(m.n_layers * m.dim * m.dim)
    m.wk = take(m.n_layers * m.dim * m.kv_dim)
    m.wv = take(m.n_layers * m.dim * m.kv_dim)
    m.wo = take(m.n_layers * m.dim * m.dim)
    m.rms_ffn = take(m.n_layers * m.dim)
    m.w1 = take(m.n_layers * m.hidden * m.dim)
    m.w2 = take(m.n_layers * m.dim * m.hidden)
    m.w3 = take(m.n_layers * m.hidden * m.dim)
    m.rms_final = take(m.dim)
    take(m.seq_len * m.head_size // 2)  # freq_cis_real (unused)
    take(m.seq_len * m.head_size // 2)  # freq_cis_imag (unused)
    m.wcls = m.emb if m.shared else take(m.vocab * m.dim)
    m.kc = [0.0] * (m.n_layers * m.seq_len * m.kv_dim)
    m.vc = [0.0] * (m.n_layers * m.seq_len * m.kv_dim)
    return m


def rmsnorm(out, x, w, off, dim):
    ss = 0.0
    for i in range(dim):
        ss += x[i] * x[i]
    s = 1.0 / math.sqrt(ss / dim + 1e-5)
    for i in range(dim):
        out[i] = w[off + i] * s * x[i]


def matmul(out, w, x, rows, cols, off):
    for r in range(rows):
        acc = 0.0
        base = off + r * cols
        for c in range(cols):
            acc += w[base + c] * x[c]
        out[r] = acc


def forward(m, token, pos):
    dim, hs = m.dim, m.head_size
    x = [0.0] * dim
    xb = [0.0] * dim
    xb2 = [0.0] * dim
    hb = [0.0] * m.hidden
    hb2 = [0.0] * m.hidden
    q = [0.0] * dim
    k = [0.0] * m.kv_dim
    v = [0.0] * m.kv_dim
    att = [0.0] * m.seq_len
    for i in range(dim):
        x[i] = m.emb[token * dim + i]
    for l in range(m.n_layers):
        rmsnorm(xb, x, m.rms_att, l * dim, dim)
        matmul(q, m.wq, xb, dim, dim, l * dim * dim)
        matmul(k, m.wk, xb, m.kv_dim, dim, l * dim * m.kv_dim)
        matmul(v, m.wv, xb, m.kv_dim, dim, l * dim * m.kv_dim)
        for h in range(m.n_heads):
            for i in range(hs // 2):
                freq = 10000.0 ** (-2.0 * i / hs)
                a = pos * freq
                c, s = math.cos(a), math.sin(a)
                i0 = h * hs + 2 * i
                i1 = i0 + 1
                v0, v1 = q[i0], q[i1]
                q[i0] = v0 * c - v1 * s
                q[i1] = v0 * s + v1 * c
            if h < m.n_kv_heads:
                for i in range(hs // 2):
                    freq = 10000.0 ** (-2.0 * i / hs)
                    a = pos * freq
                    c, s = math.cos(a), math.sin(a)
                    i0 = h * hs + 2 * i
                    i1 = i0 + 1
                    v0, v1 = k[i0], k[i1]
                    k[i0] = v0 * c - v1 * s
                    k[i1] = v0 * s + v1 * c
        kvoff = l * m.seq_len * m.kv_dim + pos * m.kv_dim
        for i in range(m.kv_dim):
            m.kc[kvoff + i] = k[i]
            m.vc[kvoff + i] = v[i]
        for h in range(m.n_heads):
            kvh = h // (m.n_heads // m.n_kv_heads)
            aoff = h * hs
            for t in range(pos + 1):
                kbase = l * m.seq_len * m.kv_dim + t * m.kv_dim + kvh * hs
                acc = 0.0
                for i in range(hs):
                    acc += q[aoff + i] * m.kc[kbase + i]
                att[t] = acc / math.sqrt(hs)
        softmax(att, pos + 1)
        for h in range(m.n_heads):
            kvh = h // (m.n_heads // m.n_kv_heads)
            aoff = h * hs
            for i in range(hs):
                acc = 0.0
                for t in range(pos + 1):
                    acc += att[t] * m.vc[l * m.seq_len * m.kv_dim
                                         + t * m.kv_dim + kvh * hs + i]
                xb[aoff + i] = acc
        matmul(xb2, m.wo, xb, dim, dim, l * dim * dim)
        for i in range(dim):
            x[i] += xb2[i]
        rmsnorm(xb, x, m.rms_ffn, l * dim, dim)
        matmul(hb, m.w1, xb, m.hidden, dim, l * m.hidden * dim)
        matmul(hb2, m.w3, xb, m.hidden, dim, l * m.hidden * dim)
        for i in range(m.hidden):
            v1 = hb[i]
            hb[i] = v1 / (1.0 + math.exp(-v1)) * hb2[i]
        matmul(xb2, m.w2, hb, dim, m.hidden, l * dim * m.hidden)
        for i in range(dim):
            x[i] += xb2[i]
    rmsnorm(x, x, m.rms_final, 0, dim)
    logits = [0.0] * m.vocab
    matmul(logits, m.wcls, x, m.vocab, dim, 0)
    return logits


def softmax(v, n):
    mx = max(v[:n])
    s = 0.0
    for i in range(n):
        v[i] = math.exp(v[i] - mx)
        s += v[i]
    for i in range(n):
        v[i] /= s


def main():
    m = load(sys.argv[1])
    toks = [int(t) for t in sys.argv[2].split(",")]
    extra = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    ids = []
    pos = 0
    for t in toks:
        logits = forward(m, t, pos)
        a = max(range(m.vocab), key=lambda i: logits[i])
        ids.append(a)
        print(f"pos {pos} argmax {a}")
        for v in logits:
            print("%.17g" % v)
        pos += 1
    while extra > 0:
        logits = forward(m, ids[-1], pos)
        a = max(range(m.vocab), key=lambda i: logits[i])
        ids.append(a)
        print(f"pos {pos} argmax {a}")
        for v in logits:
            print("%.17g" % v)
        pos += 1
        extra -= 1


if __name__ == "__main__":
    main()
