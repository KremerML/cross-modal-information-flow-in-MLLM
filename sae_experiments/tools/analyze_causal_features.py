"""Post-hoc analysis of v2 causal feature identification results.

Performs the comparisons suggested by the adversarial review:
1. Overlap between v1 (ratio) and v2 (causal) feature sets
2. Component analysis: |grad| alone vs |activation| alone vs product
3. Feature activation magnitude distributions
4. Comparison of top features across metrics
"""

import json
import sys
from pathlib import Path
import numpy as np


def load_stats(path):
    with open(path) as f:
        return json.load(f)


def top_k_set(stats, key, k=200, reverse=True):
    ranked = sorted(stats.items(), key=lambda x: x[1].get(key, 0), reverse=reverse)
    return [idx for idx, _ in ranked[:k]]


def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def main():
    base = Path("output/sae_experiments/sae_clevr_lite_layer11_attn_out_question")
    causal_dir = base.parent / (base.name + "_causal")

    v1_catalog = base / "feature_catalog.json"
    v2_stats = causal_dir / "causal_feature_stats.json"
    v2_summary = causal_dir / "causal_summary.json"

    if not v2_stats.exists():
        print(f"Causal stats not found at {v2_stats}. Run 02b first.")
        sys.exit(1)

    v2 = load_stats(v2_stats)

    print("=" * 70)
    print("POST-HOC ANALYSIS: Causal Feature Identification (v2)")
    print("=" * 70)

    if v2_summary.exists():
        summary = load_stats(v2_summary)
        print(f"\nSamples processed: {summary.get('n_processed', '?')}")
        print(f"Target: {summary.get('target', '?')}")
        print(f"Position type: {summary.get('position_type', '?')}")

    # 1. v1 vs v2 overlap
    if v1_catalog.exists():
        v1 = load_stats(v1_catalog)
        v1_set = set(v1.keys())
        v2_top = set(str(x) for x in top_k_set(v2, "causal_score", 200))
        overlap = v1_set & v2_top
        print(f"\n--- v1 (ratio) vs v2 (causal) top-200 overlap ---")
        print(f"  v1 features: {len(v1_set)}")
        print(f"  v2 features: {len(v2_top)}")
        print(f"  Overlap: {len(overlap)} ({100*len(overlap)/200:.1f}%)")
        print(f"  Jaccard: {jaccard(list(v1_set), list(v2_top)):.3f}")

    # 2. Component analysis
    print(f"\n--- Component analysis: metric comparison ---")
    by_causal = top_k_set(v2, "causal_score", 200)
    by_act = top_k_set(v2, "activation_mean", 200)
    by_grad = top_k_set(v2, "gradient_mean", 200)

    print(f"  Top-200 by causal_score vs activation_mean: "
          f"Jaccard={jaccard(by_causal, by_act):.3f}, "
          f"overlap={len(set(by_causal) & set(by_act))}")
    print(f"  Top-200 by causal_score vs gradient_mean:   "
          f"Jaccard={jaccard(by_causal, by_grad):.3f}, "
          f"overlap={len(set(by_causal) & set(by_grad))}")
    print(f"  Top-200 by activation_mean vs gradient_mean: "
          f"Jaccard={jaccard(by_act, by_grad):.3f}, "
          f"overlap={len(set(by_act) & set(by_grad))}")

    # 3. Distributions
    causals = np.array([v["causal_score"] for v in v2.values()])
    acts = np.array([v["activation_mean"] for v in v2.values()])
    grads = np.array([v["gradient_mean"] for v in v2.values()])

    nonzero_causal = causals[causals > 0]
    nonzero_act = acts[acts > 0]

    print(f"\n--- Distribution summary ---")
    print(f"  Features with causal_score > 0: {len(nonzero_causal)} / {len(causals)} "
          f"({100*len(nonzero_causal)/len(causals):.1f}%)")
    print(f"  Features with activation > 0:   {len(nonzero_act)} / {len(acts)} "
          f"({100*len(nonzero_act)/len(acts):.1f}%)")

    for name, arr in [("causal_score", causals), ("activation", acts), ("gradient", grads)]:
        if arr.max() > 0:
            print(f"  {name:15s}: p50={np.percentile(arr,50):.2e}, "
                  f"p90={np.percentile(arr,90):.2e}, "
                  f"p99={np.percentile(arr,99):.2e}, "
                  f"max={arr.max():.2e}")

    # 4. Top-20 features table
    print(f"\n--- Top-20 features by causal_score ---")
    print(f"  {'Rank':>4} {'Feature':>8} {'Causal':>12} {'Activation':>12} "
          f"{'Gradient':>12} {'in_v1_top200':>12}")
    v1_set_str = set(v1.keys()) if v1_catalog.exists() else set()
    for rank, feat_idx in enumerate(by_causal[:20], 1):
        s = v2[feat_idx]
        in_v1 = "YES" if feat_idx in v1_set_str else "no"
        print(f"  {rank:4d} {feat_idx:>8s} {s['causal_score']:12.6f} "
              f"{s['activation_mean']:12.6f} {s['gradient_mean']:12.6f} "
              f"{in_v1:>12s}")

    # 5. Correlation between components
    valid = causals > 0
    if valid.sum() > 10:
        from numpy import corrcoef
        c_a = corrcoef(causals[valid], acts[valid])[0, 1]
        c_g = corrcoef(causals[valid], grads[valid])[0, 1]
        a_g = corrcoef(acts[valid], grads[valid])[0, 1]
        print(f"\n--- Correlations (among features with causal_score > 0) ---")
        print(f"  causal vs activation: r={c_a:.3f}")
        print(f"  causal vs gradient:   r={c_g:.3f}")
        print(f"  activation vs gradient: r={a_g:.3f}")


if __name__ == "__main__":
    main()
