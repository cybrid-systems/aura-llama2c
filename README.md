# aura-llama2c

Llama-2 forward-pass inference in [Aura](https://github.com/cybrid-systems/aura)
— an Aura port of [karpathy/llama2.c](https://github.com/karpathy/llama2.c)'s
`run.c`. The project doubles as a stress test for Aura: bugs and gaps found
while porting are filed upstream and tracked in
[IMPROVEMENTS.md](IMPROVEMENTS.md).

用 [Aura](https://github.com/cybrid-systems/aura) 重写
[karpathy/llama2.c](https://github.com/karpathy/llama2.c) 的推理引擎
（`run.c`）。移植过程中发现的 Aura bug 和改进点会提到 Aura 仓库，并记录在
[IMPROVEMENTS.md](IMPROVEMENTS.md)。

## Status / 进度

| Milestone | Status |
|---|---|
| Weight export: llama2.c legacy binary layout → per-section text | ✅ `scripts/export_weights.py` |
| Forward pass: rmsnorm / matmul / RoPE (GQA) / attention+KV / silu FFN / logits | ✅ `lib/l2c/model.aura` |
| Kernel unit tests (rope, softmax, rmsnorm, matmul) — bit-exact vs float64 oracle | ✅ `tests/` |
| End-to-end logits vs pure-python float64 reference (`scripts/reference.py`) | ✅ argmax 5/5, \|dlogit\| < 1e-2 |
| Greedy generation loop | ✅ (token ids; tokenizer is Phase 2) |
| stories260K end-to-end | ⚠️ runs, but load+forward is minutes-scale on the Soft interpreter (see IMPROVEMENTS) |
| Phase 2: sentencepiece BPE tokenizer (tok512.bin) | ⬜ |
| Phase 3: temperature/top-p sampling | ⬜ |

`STATE.txt` holds the same status in key=value form.

## Running

Build Aura with an explicit build type (`RelWithDebInfo`); a plain
`cmake -B build` gives an unoptimized binary that runs the kernels ~3-5×
slower. All code runs on the Soft face:

```bash
export AURA_BIN=$HOME/code/aura/build/aura     # your aura build
export AURA_PATH=$PWD/lib:$HOME/code/aura/lib
export AURA_SANDBOX=off                        # Restricted denies vector-set!
export AURA_PIPELINE_STRICT=0                  # strict pipeline refuses named let

# tests (kernel oracles + end-to-end tiny model)
bash tests/run_tests.sh

# generate from the tiny model (token ids in/out)
L2C_WEIGHTS_DIR=model/tiny $AURA_BIN -e '(require "l2c/model") (l2c:main (list 1 8 61) 10)'
```

The tiny model (`scripts/make_tiny_model.py` → `model/tiny.bin`, 4.3K params)
is the correctness bed: it loads in seconds and the float64 reference pins
every logit. `wq`/`wk` are deliberately scaled ×20 there so attention scores
are live — with the original 0.05 scale every softmax was uniform and no RoPE
bug could show up in the logits.

The real model (tinyllamas `stories260K`, 260K params, GQA 8/4 heads,
vocab 512) converts with `scripts/export_weights.py`:

```bash
curl -L -o model/stories260K.bin \
  https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/stories260K.bin
python3 scripts/export_weights.py model/stories260K.bin --dir model/stories260K
L2C_WEIGHTS_DIR=model/stories260K $AURA_BIN -e '(require "l2c/model") (l2c:main (list 1 8 61) 2)'
```

Expect minutes per token on the Soft interpreter — the port is currently
memory/GC-bound, not FLOP-bound (see IMPROVEMENTS.md). That gap is the port's
reason to exist: every multiplier there is an Aura improvement issue.

## Layout

```
lib/l2c/model.aura      forward pass + loader + greedy sampler
scripts/export_weights.py   .bin → per-section text weights
scripts/make_tiny_model.py  seeded random tiny checkpoint (correctness bed)
scripts/reference.py        pure-python float64 reference (oracle)
tests/kernel_oracle.py      expected outputs for kernel unit tests
tests/kernels.aura          the same kernels in Aura
tests/run_tests.sh          the whole suite
```
