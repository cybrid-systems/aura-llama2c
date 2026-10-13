#!/usr/bin/env python3
"""Python oracle for the Aura sampler (Park-Miller + repeated-argmax top-p).

Mirrors lib/l2c/sampler.aura bit-for-bit. run.c uses xorshift64 over u64,
which Aura's 63-bit fixnums cannot represent, so both implementations here
use a Park-Miller LCG instead (documented divergence; aura and this oracle
agree exactly, run.c's stream differs by design).

Usage:
  python3 scripts/sampler_reference.py "0.1,2.5,-0.3" 0.8 0.9 42 5
      logits, temperature, topp, seed, n-steps -> 5 sampled ids, one per line
"""
import sys

state = [20261013]


def seed(s):
    state[0] = 1 + (s % 2147483646)


def rand():
    state[0] = (state[0] * 48271) % 2147483647
    return state[0] / 2147483647.0


def argmax(v, n):
    best, bv = 0, v[0]
    for i in range(1, n):
        if v[i] > bv:
            best, bv = i, v[i]
    return best


def softmax(v, n):
    mx = max(v[:n])
    s = 0.0
    for i in range(n):
        v[i] = 2.718281828459045 ** (v[i] - mx)
        s += v[i]
    for i in range(n):
        v[i] /= s


def sample(logits, temperature, topp):
    n = len(logits)
    if temperature <= 0.0:
        return argmax(logits, n)
    v = [logits[i] / temperature for i in range(n)]
    softmax(v, n)
    chosen = []  # (id, prob), appended in descending-prob order
    cum = 0.0
    while cum < topp:
        i = argmax(v, n)
        p = v[i]
        v[i] = -1.0
        chosen.append((i, p))
        cum += p
    total = sum(p for _, p in chosen)
    r = rand() * total
    acc = 0.0
    for i, p in chosen:
        acc += p
        if r < acc:
            return i
    return chosen[-1][0]


def main():
    logits = [float(x) for x in sys.argv[1].split(",")]
    temperature = float(sys.argv[2])
    topp = float(sys.argv[3])
    seed(int(sys.argv[4]))
    steps = int(sys.argv[5])
    for _ in range(steps):
        print(sample(logits, temperature, topp))


if __name__ == "__main__":
    main()
