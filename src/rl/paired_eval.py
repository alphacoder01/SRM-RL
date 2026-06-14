"""Paired comparison of two RL eval runs (e.g. RL-tuned vs pretrained baseline).

Both runs must be eval-only with `rl.eval.dump_samples=true` and the SAME
`rl.eval.num_samples` / `rl.eval.num_fill`, so the per-sample index keys refer
to identical puzzles (deterministic seeding). Because the comparison is paired,
a McNemar test on the discordant puzzles is far more powerful than comparing
marginal accuracies (cf. Decisions.md D25).

Usage:
    python -m src.rl.paired_eval <run_a>/eval_samples_it0.jsonl <run_b>/eval_samples_it0.jsonl
    # or pass run folders; the latest eval_samples_*.jsonl in each is used
"""
from __future__ import annotations

import json
import sys
from math import comb, sqrt
from pathlib import Path


def load(path: Path) -> dict[int, int]:
    if path.is_dir():
        candidates = sorted(path.glob("eval_samples_it*.jsonl"))
        if not candidates:
            raise FileNotFoundError(f"no eval_samples_*.jsonl in {path}")
        path = candidates[-1]
    out = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                out[r["index"]] = int(r["correct"])
    return out, path


def normal_cdf(x: float) -> float:
    # via erf
    from math import erf
    return 0.5 * (1 + erf(x / sqrt(2)))


def mcnemar_p(b: int, c: int) -> tuple[float, float]:
    """Exact two-sided McNemar p-value (binomial) and the continuity-corrected
    chi-square z. b = A-correct & B-wrong, c = A-wrong & B-correct."""
    n = b + c
    if n == 0:
        return 1.0, 0.0
    k = min(b, c)
    # exact two-sided binomial test against p=0.5
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    p_exact = min(1.0, 2 * tail)
    z = (abs(b - c) - 1) / sqrt(n) if n > 0 else 0.0
    return p_exact, z


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    a, pa = load(Path(sys.argv[1]))
    b, pb = load(Path(sys.argv[2]))
    common = sorted(set(a) & set(b))
    if not common:
        raise ValueError("no overlapping sample indices between the two runs")

    n = len(common)
    acc_a = sum(a[i] for i in common) / n
    acc_b = sum(b[i] for i in common) / n
    # discordant pairs
    a_right_b_wrong = sum(1 for i in common if a[i] and not b[i])
    a_wrong_b_right = sum(1 for i in common if not a[i] and b[i])
    both = sum(1 for i in common if a[i] and b[i])
    neither = sum(1 for i in common if not a[i] and not b[i])

    p_exact, z = mcnemar_p(a_right_b_wrong, a_wrong_b_right)

    print(f"Run A: {pa}")
    print(f"Run B: {pb}")
    print(f"\nPaired on {n} common puzzles")
    print(f"  acc(A) = {acc_a:.4f}   acc(B) = {acc_b:.4f}   diff = {acc_a - acc_b:+.4f}")
    print(f"  both correct: {both}   neither: {neither}")
    print(f"  A>B (A right, B wrong): {a_right_b_wrong}")
    print(f"  B>A (B right, A wrong): {a_wrong_b_right}")
    print(f"\nMcNemar (paired) two-sided exact p = {p_exact:.4f}   "
          f"(continuity-corrected z = {z:.2f})")
    verdict = "significant" if p_exact < 0.05 else "NOT significant"
    print(f"  -> difference is {verdict} at alpha=0.05")
    if p_exact >= 0.05 and a_right_b_wrong + a_wrong_b_right > 0:
        # power hint: how many more discordant pairs of the same ratio would be needed
        print("  (inconclusive — collect more eval samples to increase power)")


if __name__ == "__main__":
    main()
