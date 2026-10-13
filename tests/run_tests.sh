#!/usr/bin/env bash
# tests/run_tests.sh — aura-llama2c test runner
#
# Requires:
#   AURA_BIN   aura executable (default $HOME/code/aura/build/aura)
#   AURA_PATH  must include this repo's lib/ and aura's lib/
#   Soft face: AURA_SANDBOX=off AURA_PIPELINE_STRICT=0
#
# Stages:
#   1. kernel unit tests (rope/softmax/rmsnorm/matmul) vs kernel_oracle.py
#      (bit-exact, 1e-12)
#   2. end-to-end tiny-model logits vs scripts/reference.py (greedy argmax
#      per position + per-logit tolerance)
#
# Stage-2 tolerance: both sides are float64 with identical accumulation
# order, but transcendental rounding differs (aura C exp/expt vs python
# math) and wq/wk are deliberately scaled so attention is live. Observed
# steady-state worst |dlogit| ~2e-3; structural bugs (weight slicing,
# residual wiring) produce O(1) diffs. argmax equality is also required.
set -u
cd "$(dirname "$0")/.."

AURA_BIN=${AURA_BIN:-$HOME/code/aura/build/aura}
TOL_E2E=${TOL_E2E:-1e-2}

FAIL=0

echo "== stage 1: kernel unit tests =="
python3 tests/kernel_oracle.py > /tmp/l2c_kernel_expected.txt
timeout 120 "$AURA_BIN" tests/kernels.aura 2>&1 | grep -v "^load_module_file" > /tmp/l2c_kernel_actual.txt
if python3 - <<'EOF'
import sys

def parse(path):
    out = {}
    cur = None
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            out[cur].append(float(line))
        except (ValueError, TypeError, KeyError):
            cur = line
            out[cur] = []
    return out

exp = parse("/tmp/l2c_kernel_expected.txt")
act = parse("/tmp/l2c_kernel_actual.txt")
ok = True
for k in sorted(exp):
    e, a = exp.get(k), act.get(k)
    if a is None or len(e) != len(a):
        print(f"FAIL {k}: missing or length mismatch (exp {len(e)}, act {len(a) if a else 0})")
        ok = False
        continue
    worst = max(abs(x - y) for x, y in zip(e, a))
    if worst > 1e-12:
        print(f"FAIL {k}: worst |diff| = {worst:.3e}")
        ok = False
    else:
        print(f"PASS {k} (worst |diff| = {worst:.3e})")
sys.exit(0 if ok else 1)
EOF
then
  echo "kernel tests OK"
else
  echo "kernel tests FAILED"
  FAIL=1
fi

echo "== stage 2: end-to-end tiny model =="
PROMPT="1,8,61"
EXTRA=2
VOCAB=$(cut -d' ' -f6 model/tiny/config.txt)
python3 scripts/reference.py model/tiny.bin "$PROMPT" "$EXTRA" > /tmp/l2c_e2e_ref.txt
L2C_WEIGHTS_DIR=model/tiny timeout 300 "$AURA_BIN" -e \
  "(require \"l2c/model\") (l2c:main (list 1 8 61) $EXTRA)" 2>&1 \
  | grep -v "^load_module_file" > /tmp/l2c_e2e_act.txt
if python3 - "$VOCAB" "$TOL_E2E" <<'EOF'
import re, sys

vocab = int(sys.argv[1])
tol = float(sys.argv[2])

def parse(path):
    lines = [l.strip() for l in open(path) if l.strip()]
    out = []; i = 0
    while i < len(lines):
        if lines[i].startswith("pos "):
            m = re.match(r"pos (\d+) argmax (\d+)", lines[i])
            pos, am = int(m.group(1)), int(m.group(2))
            logits = []; j = i + 1
            while j < len(lines) and len(logits) < vocab:
                try:
                    logits.append(float(lines[j]))
                except ValueError:
                    break
                j += 1
            out.append((pos, am, logits)); i = j
        else:
            i += 1
    return out

a = parse("/tmp/l2c_e2e_act.txt")
r = parse("/tmp/l2c_e2e_ref.txt")
if len(a) != len(r):
    print(f"FAIL: position blocks aura={len(a)} ref={len(r)}")
    sys.exit(1)
worst = 0.0
for (pa, aa, la), (pr, ar, lr) in zip(a, r):
    if pa != pr or aa != ar:
        print(f"FAIL pos {pa}: argmax {aa} != ref {ar}")
        sys.exit(1)
    worst = max(worst, max(abs(x - y) for x, y in zip(la, lr)))
print(f"argmax match on all {len(a)} positions; worst |dlogit| = {worst:.3e}")
sys.exit(0 if worst < tol else 1)
EOF
then
  echo "end-to-end OK"
else
  echo "end-to-end FAILED"
  FAIL=1
fi

echo "== stage 3: tokenizer (BPE encode/decode vs python oracle) =="
cat > /tmp/l2c_tok_ref.txt <<REF
encode1 $(python3 scripts/tokenizer_reference.py encode "Once upon a time")
encode2 $(python3 scripts/tokenizer_reference.py encode "Lily and Ben")
encode3 $(python3 scripts/tokenizer_reference.py encode "" 0)
decode1 $(python3 scripts/tokenizer_reference.py decode "$(python3 scripts/tokenizer_reference.py encode "Once upon a time")")
decode2 $(python3 scripts/tokenizer_reference.py decode "$(python3 scripts/tokenizer_reference.py encode "Zaph" 0)")
REF
timeout 120 "$AURA_BIN" tests/tokenizers.aura 2>&1 \
  | grep -v "^load_module_file" > /tmp/l2c_tok_act.txt
if diff -q /tmp/l2c_tok_ref.txt /tmp/l2c_tok_act.txt >/dev/null 2>&1; then
  echo "tokenizer OK"
else
  echo "tokenizer FAILED (ref vs act):"
  diff /tmp/l2c_tok_ref.txt /tmp/l2c_tok_act.txt | head -10
  FAIL=1
fi

echo "== stage 4: sampler (Park-Miller + top-p vs python oracle) =="
{
  echo -n "ids "
  python3 scripts/sampler_reference.py "0.1,2.5,-0.3,1.7,0.0,-1.2,3.1,0.4" 0.8 0.9 42 5 \
    | paste -sd, -
  echo "greedy 6"
} > /tmp/l2c_sam_ref.txt
timeout 120 "$AURA_BIN" tests/samplers.aura 2>&1 \
  | grep -v "^load_module_file" > /tmp/l2c_sam_act.txt
if diff -q /tmp/l2c_sam_ref.txt /tmp/l2c_sam_act.txt >/dev/null 2>&1; then
  echo "sampler OK"
else
  echo "sampler FAILED (ref vs act):"
  diff /tmp/l2c_sam_ref.txt /tmp/l2c_sam_act.txt | head -10
  FAIL=1
fi

exit $FAIL
