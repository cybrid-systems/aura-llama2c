# Aura findings from the llama2.c port

Aura tip at port time: `45c8bf425` (2026-10-13). Face: Soft
(`AURA_SANDBOX=off AURA_PIPELINE_STRICT=0`), `AURA_PATH` including aura's
`lib/`, RelWithDebInfo build. Each finding has a minimal repro and an
upstream issue.

## Filed upstream

| Issue | Finding | Hit while |
|---|---|---|
| [#4500](https://github.com/cybrid-systems/aura/issues/4500) | `lib/std/encoding.aura` hex/base64 wrappers hang forever on any input: `(define (hex-encode d) (encoding:hex-encode d))` — the colon-qualified name resolves to the module's own export (the wrapper itself), not the native prim registered as `encoding:hex-encode` in `evaluator_primitives_char.cpp:308`. Infinite tail recursion, no depth-guard trip. All of hex-encode/hex-decode/base64-encode/base64-decode are dead on HEAD. | binary model loading plan (hex path), first probe |
| [#4496](https://github.com/cybrid-systems/aura/issues/4496) | std missing common Scheme bindings: assoc/assq/iota/case/do/vector-fill!/vector-map/vector-copy/string->symbol/symbol->string/let-values/exit (exit also can't stop the program) | loader/section walking wanted iota+assoc; test runners want exit |
| [#4497](https://github.com/cybrid-systems/aura/issues/4497) | `vector-ref` out-of-range returns `<unknown>`, execution continues, exit 0 — silent data poisoning for index math (this port is ALL index math) | forward pass index arithmetic |
| [#4498](https://github.com/cybrid-systems/aura/issues/4498) | reader: `.5`/`1.` rejected as parse errors; `1/2` silently reads as `(1 /2 3)` — two tokens, no diagnostic | transcribing weight constants / formulas |
| [#4499](https://github.com/cybrid-systems/aura/issues/4499) | `(/ 7 2)` → 3 (int truncation): every float division site needs a manual `(* 1.0 x)` or literal floats; port writes `(* 1.0 ...)` guards everywhere | scaling factors (attention /sqrt(hs), rmsnorm mean) |

## Unfiled (measurement first / design)

- **Unboxed float vectors (the headline).** Every `vector-ref`/`vector-set!`
  of a float boxes/unboxes; every `+`/`*` allocates an intermediate boxed
  double. Combined with the file-mode partial reclaim (#4474 family: whole
  live heap marked every 128 allocations), loading 260K float weights
  through text did not finish in 30 minutes of wall time, and the 21K-param
  tiny model costs ~30-60 s per token (GC-bound, not FLOP-bound). The C
  reference does 260K MACs in ~0.3 ms. A typed `f64vector` primitive (or
  boxed-value elision in numeric loops) would collapse both — filed as
  [#4501](https://github.com/cybrid-systems/aura/issues/4501).
- **`file->lines` on multi-MB text is the loader bottleneck.** Splitting
  3.4 MB into 260K strings holds the whole list live while parsing; sections
  are now split into per-section files to bound live-heap growth. A
  streaming line-reader primitive (or lazy lines) would remove the loader
  workaround entirely.
- **`string->number` throughput** on 260K single-float lines is a visible
  slice of load time; a `read-f64vector`-style bulk primitive would remove
  text parsing altogether (binary read is 8-bit clean today, but there is
  no way to reinterpret 4 bytes as float32 — no bit-cast/struct primitive).
- **`setenv`/`getenv`**: `getenv` exists, `setenv` does not (driver env
  selection currently goes through the shell).
- **Deep recursion is depth-capped (#4454) so all loops are written as tail
  calls** — fine, but a `do`/`for` looping form (#4496) would make hot
  kernels read like run.c and drop the named-let boilerplate.

## Port-side lessons (our bugs, not aura's)

- **#4499 bit its own port:** `(/ sam:state 2147483647)` on two exact ints
  truncated to 0 forever — the sampler always picked the top-probability
  token and the top-p test caught it as a constant-output failure. Fixed by
  `(* 1.0 sam:state)` before dividing.
- **Run.c tokenizer convention:** sentencepiece U+2581 must be exported as a
  literal space; otherwise run.c's dummy-prefix `str_lookup(" ")` returns -1
  and run.c segfaults on `vocab[-1]`. This is a run.c/exporter contract, not
  an aura bug.
- **Random-weight end-to-end oracles are insensitive:** with scale-0.05
  weights the logits are dominated by the `wcls·emb` direct path; RoPE was
  inverted end-to-end and the logits still matched to 1e-6. Kernel-level
  oracles (fixed inputs, exact outputs) are the only sharp validation for
  positional/attention machinery.
- **Greedy generation on a random model diverges at near-ties by design:**
  per-position logit sums match the float64 reference to ~1e-9 at pos 0,
  growing to ~1e-2 by pos 11 (FMA contraction + libm paths + %.9g parse
  rounding, amplified through layers). On an untrained model the logits are
  near-degenerate, so an argmax over a 1e-3-level gap flips on noise (seen:
  id 187 vs 361 at generation step 8, top-2 gap 0.0012). The full-chain id
  comparison matched 12/13; the flip is numerical, not semantic. Trained
  checkpoints (stories260K) have decisive logits and don't exhibit this.
- **Aura byte-fallback rendering:** run.c emits raw bytes for byte pieces;
  aura strings are code-point sequences, so bytes ≥ 0x80 render as UTF-8
  encodings of their code points instead. Identical on ASCII (TinyStories
  domain); documented divergence otherwise.

## Validation notes (for future ports)

- Kernel-level oracles (fixed input vectors → expected %.17g output,
  compared at 1e-12) catch what end-to-end logits cannot: with random
  weights the logits are dominated by the `wcls·emb` direct path —
  attention/RoPE perturbations sit at 1e-6 even when attention weights are
  scaled ×20. End-to-end argmax equality is a smoke test, not a proof.
- The end-to-end float64 reference must mirror accumulation order to stay
  near 1e-6; residual ~2e-3 comes from transcendental rounding paths
  (C exp/expt vs python math) amplified by live attention scores.
