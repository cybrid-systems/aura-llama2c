#!/usr/bin/env python3
"""Generate kernel-level expected outputs for the Aura port's unit tests.

Prints test-vector inputs and the expected outputs computed here in
float64 with the run.c formulas. tests/kernels.aura runs the same kernels
through the Aura implementation and prints in the same layout;
tests/run_tests.sh diffs the two.

Layout (all %.17g):
  rope:  8 floats (one head, head_size 8, pos 3)
  softmax: 4 floats from [1.2, -0.3, 0.0, 2.7]
  rmsnorm: 8 floats, x = [0.5,-1.25,2,0,3.5,-0.75,1.5,-2], w = 0.1+j*0.01
  matmul: 3 floats, W 3x2 row-major [1,-2,3,0.5,-1,2.25] @ [2,-1]
"""
import math

hs = 8
pos = 3
x = [1.0, 0.0, -2.0, 0.5, 3.0, -1.5, 0.25, 2.0]  # rope input
xr = list(x)
for i in range(hs // 2):
    freq = 10000.0 ** (-(2.0 * i) / hs)
    a = pos * freq
    c, s = math.cos(a), math.sin(a)
    i0, i1 = 2 * i, 2 * i + 1
    v0, v1 = xr[i0], xr[i1]
    xr[i0] = v0 * c - v1 * s
    xr[i1] = v0 * s + v1 * c

sv = [1.2, -0.3, 0.0, 2.7]
mx = max(sv)
se = [math.exp(v - mx) for v in sv]
ss = sum(se)
sm = [v / ss for v in se]

rx = [0.5, -1.25, 2.0, 0.0, 3.5, -0.75, 1.5, -2.0]
rw = [0.1 + j * 0.01 for j in range(8)]
rss = sum(v * v for v in rx)
s = 1.0 / math.sqrt(rss / 8 + 1e-5)
rm = [rw[j] * s * rx[j] for j in range(8)]

W = [1.0, -2.0, 3.0, 0.5, -1.0, 2.25]
v = [2.0, -1.0]
mm = [sum(W[r * 2 + c] * v[c] for c in range(2)) for r in range(3)]

print("rope")
for f in xr:
    print("%.17g" % f)
print("softmax")
for f in sm:
    print("%.17g" % f)
print("rmsnorm")
for f in rm:
    print("%.17g" % f)
print("matmul")
for f in mm:
    print("%.17g" % f)
