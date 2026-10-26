# Paper constructions

This directory contains the nine **Ours (w/ TTME)** constructions reported in
Tables 1 and 2 of the [paper](../paper.pdf).
Each JSON file preserves the saved array at full precision. The table below uses
the paper's six-decimal display precision.

| Framework | Model | Task | Paper score | Size | Construction |
|---|---|---|---:|---:|---|
| TTT-Discover | gpt-oss-120b | AC1 ↓ | 1.502745 | 32,000 | [sequence](ac1/sequence.json) |
| TTT-Discover | gpt-oss-120b | AC2 ↑ | 0.960784 | 65,536 | [sequence](ac2/sequence.json) |
| TTT-Discover | gpt-oss-120b | Erdős ↓ | 0.380862 | 801 | [sequence](erdos/sequence.json) |
| TTT-Discover | Qwen3-8B | AC1 ↓ | 1.504329 | 1,000 | [sequence](qwen3-8b/ac1/sequence.json) |
| TTT-Discover | Qwen3-8B | AC2 ↑ | 0.949248 | 4,096 | [sequence](qwen3-8b/ac2/sequence.json) |
| TTT-Discover | Qwen3-8B | Erdős ↓ | 0.380927 | 200 | [sequence](qwen3-8b/erdos/sequence.json) |
| ThetaEvolve | Distill-Qwen3-8B | AC2 ↑ | 0.948148 | 4,096 | [sequence](thetaevolve/ac2/sequence.json) |
| ThetaEvolve | Distill-Qwen3-8B | AC3 ↓ | 1.464224 | 100 | [sequence](thetaevolve/ac3/sequence.json) |
| ThetaEvolve | Distill-Qwen3-8B | Hadamard ↑ | 0.579938 | 29 × 29 | [matrix](thetaevolve/hadamard/matrix.json) |

Distill-Qwen3-8B denotes DeepSeek-R1-0528-Qwen3-8B. The TTT-Discover results
use a 50-step search budget; the ThetaEvolve results use 65 steps. The stored
gpt-oss-120b AC2 array is the admitted step-49 construction, with score
`0.9607839870598152` and length 65,536.

## Verify

After installing the package, run this command from the repository root:

```bash
python scripts/verify_results.py
```

Verification needs only a CPU and NumPy. It checks each file's SHA-256, the exact
array values, shape, validity, archive length limits where applicable, discovery
step, recomputed score and agreement with the paper at six decimal places.
Floating-point scores must agree with the manifest within `1e-12` absolute or
relative tolerance. The Hadamard determinant is computed with integer arithmetic:
its absolute value is `736926518386354028544`, and its score divides that value by
`2**28 * 7**12 * 342`.

The [manifest](manifest.json) records full-precision scores, paper table numbers,
zero-based discovery steps and file hashes. Array hashes use row-major,
little-endian float64 bytes for sequences and signed int8 bytes for the matrix.
The matrix retains its original row and column order.

For a single TTT-Discover construction:

```bash
ttt score ac1 results/ac1/sequence.json
ttt score ac2 results/ac2/sequence.json
ttt score erdos results/erdos/sequence.json
```

## Generator snapshots

The `program.py` files next to the three gpt-oss-120b arrays are the corresponding
generated program snapshots. They use the task harness's injected parent and
verifier, and may depend on random choices or wall-clock budgets. Use the stored
arrays to reproduce the reported scores; rerunning an optimizer can return a
different construction. The manifest hashes each included program snapshot.

This collection covers the main TTME rows. Baseline and ablation constructions,
intermediate candidates and results beyond the reported search budgets are not
included. The ThetaEvolve arrays can be verified here; its search runner is not
part of this release.
