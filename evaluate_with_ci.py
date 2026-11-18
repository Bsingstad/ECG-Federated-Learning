#!/usr/bin/env python

# Computes all test-set metrics, bootstrap 95% CIs and paired comparisons for the
# revision, from the predictions saved by run_strategy.py:
#
#   python evaluate_with_ci.py -r results_revision -o tables --seeds 1 2 3 4 5
#
# Optional: --meta ptbxl_meta.csv (columns: sex, age; one row per PTB-XL recording,
# same order as y_data_ptbxl.npy; written by dataset_stats.py) for subgroup results.
#
# Bootstrap (two-level, primary): each resample draws (1) PTB-XL recordings with
# replacement (all 30 labels of a recording stay together) and (2), separately for
# each strategy, training seeds with replacement; the metric is averaged over the
# drawn seeds. The CIs and tests therefore include both test-set sampling and
# training (run-to-run) variability. The recording resamples are the same for every
# strategy, so differences between strategies are paired on the test set; seeds
# are independent training runs and are resampled independently per strategy.
# With k seeds, resampling seeds understates the training variance by the factor
# (k-1)/k, so the intervals are, if anything, still slightly too narrow.
#
# The recording-only bootstrap (training runs fixed, as in a single-model
# evaluation) is also reported (columns *_test) as a sensitivity analysis.
#
# Resampling is implemented as integer weights on the original recordings, which
# lets every metric be computed in O(N) from a sort order computed once.

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.metrics import precision_recall_curve, roc_curve

STRATEGIES = {
    "cds": "CDS",
    "ciil_1epoch": "CIIL 1 epoch",
    "iil": "IIL",
    "fedavg": "FedAvg",
    "fedavg_repeated": "Repeated FedAvg",
    "ciil": "CIIL",
    "fedsgd": "FedSGD",
}
METRICS = ["micro_auroc", "macro_auroc", "micro_auprc", "macro_auprc",
           "macro_f1", "macro_sens", "macro_spec", "macro_brier"]


def get_parser():
    parser = argparse.ArgumentParser(description="Bootstrap CIs and paired tests on PTB-XL.")
    parser.add_argument("-r", "--results_folder", type=str, required=True)
    parser.add_argument("-o", "--out_folder", type=str, default="tables")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--strategies", type=str, nargs="+", default=list(STRATEGIES))
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--boot_seed", type=int, default=2024)
    parser.add_argument("--n_jobs", type=int, default=1, help="processes (one strategy per process)")
    parser.add_argument("--meta", type=str, default=None)
    parser.add_argument("--no_delong", action="store_true", help="skip the paired DeLong tests")
    parser.add_argument("--class_names", type=str, default=None,
                        help="dx_mapping_scored.csv, to label the per-class table")
    parser.add_argument("--classes", type=str, default=None,
                        help="classes.csv from prepare_data.py (column order of y); default: sorted mapping codes")
    return parser


class SortedScores:
    """Scores sorted once, so weighted AUROC and AP can be computed in O(N)."""

    def __init__(self, y, s):
        order = np.argsort(s, kind="mergesort")
        self.order = order
        self.y = y[order].astype(np.float64)
        s_sorted = s[order]
        # start index of each group of tied scores (ascending)
        self.starts = np.flatnonzero(np.r_[True, s_sorted[1:] != s_sorted[:-1]])

    def auroc(self, w):
        w = w[self.order]
        pos = np.add.reduceat(w * self.y, self.starts)
        neg = np.add.reduceat(w * (1.0 - self.y), self.starts)
        n_pos, n_neg = pos.sum(), neg.sum()
        if n_pos == 0 or n_neg == 0:
            return np.nan
        neg_below = np.cumsum(neg) - neg
        return float(np.sum(pos * (neg_below + 0.5 * neg)) / (n_pos * n_neg))

    def auprc(self, w):
        # Average precision, as in sklearn: sum_k (R_k - R_{k-1}) P_k over
        # thresholds in decreasing order of score.
        w = w[self.order]
        pos = np.add.reduceat(w * self.y, self.starts)[::-1]
        tot = np.add.reduceat(w, self.starts)[::-1]
        n_pos = pos.sum()
        if n_pos == 0:
            return np.nan
        tp = np.cumsum(pos)
        predicted = np.cumsum(tot)
        valid = predicted > 0
        precision = np.where(valid, tp / np.where(valid, predicted, 1), 0.0)
        recall = tp / n_pos
        return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


class RunMetrics:
    """All metrics for one strategy/seed, evaluated for any recording weights."""

    def __init__(self, y_true, y_pred, thresholds):
        self.y = y_true.astype(np.float64)
        self.p = y_pred.astype(np.float64)
        self.n, self.k = y_true.shape
        self.classes = [j for j in range(self.k) if 0 < y_true[:, j].sum() < self.n]
        self.micro = SortedScores(y_true.ravel(), y_pred.ravel())
        self.per_class = {j: SortedScores(y_true[:, j], y_pred[:, j]) for j in self.classes}
        self.yhat = (y_pred >= thresholds[None, :]).astype(np.float64)

    def compute(self, w):
        w_pairs = np.repeat(w, self.k)
        out = {"micro_auroc": self.micro.auroc(w_pairs), "micro_auprc": self.micro.auprc(w_pairs)}
        aurocs = [self.per_class[j].auroc(w) for j in self.classes]
        auprcs = [self.per_class[j].auprc(w) for j in self.classes]
        out["macro_auroc"] = np.nanmean(aurocs)
        out["macro_auprc"] = np.nanmean(auprcs)
        c = self.classes
        y, yh, p = self.y[:, c], self.yhat[:, c], self.p[:, c]
        wc = w[:, None]
        tp = (wc * y * yh).sum(0)
        fp = (wc * (1 - y) * yh).sum(0)
        fn = (wc * y * (1 - yh)).sum(0)
        tn = (wc * (1 - y) * (1 - yh)).sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out["macro_f1"] = np.nanmean(2 * tp / (2 * tp + fp + fn))
            out["macro_sens"] = np.nanmean(tp / (tp + fn))
            out["macro_spec"] = np.nanmean(tn / (tn + fp))
        out["macro_brier"] = float(np.mean((wc * (p - y) ** 2).sum(0) / w.sum()))
        out["per_class_auroc"] = np.array(aurocs)
        out["per_class_auprc"] = np.array(auprcs)
        return out


def f1_thresholds(val_true, val_pred):
    """Per-class threshold maximising F1 on the pooled validation data."""
    thr = np.full(val_true.shape[1], 0.5)
    for j in range(val_true.shape[1]):
        if val_true[:, j].sum() == 0:
            continue
        precision, recall, t = precision_recall_curve(val_true[:, j], val_pred[:, j])
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
        thr[j] = t[np.argmax(f1[:-1])]
    return thr


def holm(pvals):
    """Holm adjustment. NaN p-values (test not possible) stay NaN and are not counted."""
    pvals = np.asarray(pvals, dtype=np.float64)
    adjusted = np.full_like(pvals, np.nan)
    finite = np.flatnonzero(~np.isnan(pvals))
    order = finite[np.argsort(pvals[finite])]
    running_max = 0.0
    m = len(order)
    for rank, idx in enumerate(order):
        running_max = max(running_max, (m - rank) * pvals[idx])
        adjusted[idx] = min(1.0, running_max)
    return adjusted


def delong_components(y, s):
    """AUROC and DeLong structural components (Sun & Xu 2014 fast algorithm)."""
    pos = y == 1
    x, z = s[pos], s[~pos]
    m, n = len(x), len(z)
    tx, tz = rankdata(x), rankdata(z)
    t_all = rankdata(np.r_[x, z])
    auc = (t_all[:m].sum() - m * (m + 1) / 2) / (m * n)
    v10 = (t_all[:m] - tx) / n          # one per positive
    v01 = 1.0 - (t_all[m:] - tz) / m    # one per negative
    return auc, v10, v01


def delong_paired(y, s1, s2):
    """Paired two-sided DeLong test of AUROC(s1) vs AUROC(s2) on the same cases.
    Returns (auc1, auc2, diff, se_diff, z, p)."""
    a1, v10_1, v01_1 = delong_components(y, s1)
    a2, v10_2, v01_2 = delong_components(y, s2)
    m, n = len(v10_1), len(v01_1)
    if m < 2 or n < 2:  # variance not estimable (fewer than 2 positives or negatives)
        return a1, a2, a1 - a2, np.nan, np.nan, np.nan
    s10 = np.cov(np.vstack([v10_1, v10_2]))
    s01 = np.cov(np.vstack([v01_1, v01_2]))
    cov = s10 / m + s01 / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    diff = a1 - a2
    if var <= 0:
        return a1, a2, diff, 0.0, np.nan, (1.0 if diff == 0 else 0.0)
    se = np.sqrt(var)
    zstat = diff / se
    return a1, a2, diff, se, zstat, float(2 * norm.sf(abs(zstat)))


def run_delong(y_true, preds, order, classes, seeds, out_folder):
    """Paired DeLong tests for every pair of strategies, separately for each seed.
    Micro-AUROC: all 30 labels pooled. This ignores that labels from the same
    recording are correlated, so its SE is too small (p too low); the
    recording-level bootstrap remains the primary test. Per-class AUROC: one
    observation per recording, so the DeLong independence assumption holds.
    Holm correction: over strategy pairs (micro), and over pairs x classes
    (per class), within each seed."""
    micro_rows, class_rows = [], []
    for si, seed in enumerate(seeds):
        seed_micro, seed_class = [], []
        for i, a in enumerate(order):
            for b in order[i + 1:]:
                pa, pb = preds[a][si], preds[b][si]
                a1, a2, d, se, zs, p = delong_paired(y_true.ravel(), pb.ravel(), pa.ravel())
                seed_micro.append({"seed": seed, "strategy": b, "comparator": a, "auroc": a1,
                                   "auroc_comparator": a2, "delta": d, "se": se, "z": zs, "p": p})
                for j in classes:
                    a1, a2, d, se, zs, p = delong_paired(y_true[:, j], pb[:, j], pa[:, j])
                    seed_class.append({"seed": seed, "class": j, "strategy": b, "comparator": a,
                                       "auroc": a1, "auroc_comparator": a2, "delta": d,
                                       "se": se, "z": zs, "p": p})
        seed_micro, seed_class = pd.DataFrame(seed_micro), pd.DataFrame(seed_class)
        seed_micro["p_holm"] = holm(seed_micro["p"].values)
        seed_class["p_holm"] = holm(seed_class["p"].values)
        micro_rows.append(seed_micro)
        class_rows.append(seed_class)
    micro, per_class = pd.concat(micro_rows), pd.concat(class_rows)
    micro.to_csv(os.path.join(out_folder, "results_delong_micro.csv"), index=False)
    per_class.to_csv(os.path.join(out_folder, "results_delong_per_class.csv"), index=False)

    # Summary per pair across seeds: number of seeds with Holm p < 0.05 and median p.
    summary = (micro.groupby(["strategy", "comparator"], sort=False)
               .agg(delta_mean=("delta", "mean"), p_holm_median=("p_holm", "median"),
                    p_holm_max=("p_holm", "max"),
                    n_seeds_sig=("p_holm", lambda p: int((p < 0.05).sum())))
               .reset_index())
    cls_sig = (per_class.assign(sig=per_class["p_holm"] < 0.05)
               .groupby(["strategy", "comparator", "seed"], sort=False)["sig"].sum()
               .groupby(["strategy", "comparator"], sort=False).median()
               .rename("median_n_classes_sig").reset_index())
    summary = summary.merge(cls_sig, on=["strategy", "comparator"])
    summary.to_csv(os.path.join(out_folder, "results_delong_summary.csv"), index=False)
    n_seeds = len(seeds)
    with open(os.path.join(out_folder, "results_delong_table.tex"), "w") as f:
        for r in summary.itertuples():
            f.write(f"{STRATEGIES[r.strategy]} & {STRATEGIES[r.comparator]} & "
                    f"{r.delta_mean:.3f}".replace("-", "$-$") +
                    f" & {r.p_holm_median:.3g} & {r.n_seeds_sig}/{n_seeds} & "
                    f"{r.median_n_classes_sig:g}/{len(classes)} \\\\\n")
    print(summary.to_string(index=False))


def fmt(est, lo, hi, digits=3):
    num = lambda x: f"{x:.{digits}f}".replace("-", "$-$")
    sep = "--" if lo >= 0 else " to "
    return f"{num(est)} ({num(lo)}{sep}{num(hi)})"


def bootstrap_strategy(strategy_runs, n, n_boot, boot_seed):
    """Bootstrap distribution of every metric for each seed of one strategy:
    {metric: array (n_boot, n_seeds)}. Every call draws the same recording
    resamples (same boot_seed), so results from different strategies, possibly
    computed in different processes, are paired on the test set."""
    rng = np.random.default_rng(boot_seed)
    out = {m: np.empty((n_boot, len(strategy_runs))) for m in METRICS}
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64)
        vals = [r.compute(w) for r in strategy_runs]
        for m in METRICS:
            out[m][b] = [v[m] for v in vals]
    return out


def resample_seeds(per_seed, boot_seed, strategy_id):
    """Two-level bootstrap: for each recording resample b, draw the seeds of this
    strategy with replacement and average. per_seed: {metric: (n_boot, n_seeds)}."""
    n_boot, k = next(iter(per_seed.values())).shape
    rng = np.random.default_rng([boot_seed, 1 + strategy_id])
    idx = rng.integers(0, k, (n_boot, k))
    rows = np.arange(n_boot)[:, None]
    return {m: v[rows, idx].mean(axis=1) for m, v in per_seed.items()}


def boot_p(d):
    """Two-sided bootstrap p-value for H0: difference = 0."""
    return min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0)))


def fmt_p(p, n_boot):
    return f"$<${1 / n_boot:.4f}" if p < 1 / n_boot else f"{p:.3f}"


def downsample_roc(fpr, tpr, n_points=1000):
    idx = np.unique(np.linspace(0, len(fpr) - 1, n_points).astype(int))
    return fpr[idx], tpr[idx]


def run(args):
    os.makedirs(args.out_folder, exist_ok=True)
    y_true = np.load(os.path.join(args.results_folder, "ptbxl_true.npy"))
    n = y_true.shape[0]

    runs, preds = {}, {}
    for s in args.strategies:
        runs[s], preds[s] = [], []
        for seed in args.seeds:
            d = os.path.join(args.results_folder, f"{s}_seed{seed}")
            thr = f1_thresholds(np.load(os.path.join(d, "val_true.npy")), np.load(os.path.join(d, "val_pred.npy")))
            pred = np.load(os.path.join(d, "ptbxl_pred.npy"))
            runs[s].append(RunMetrics(y_true, pred, thr))
            preds[s].append(pred)
            if seed == args.seeds[0]:
                fpr, tpr, _ = roc_curve(y_true.ravel(), pred.ravel())
                fpr, tpr = downsample_roc(fpr, tpr)
                pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(
                    os.path.join(args.out_folder, f"roc_{s}_dwnsmp.csv"), index=False)
    classes = runs[args.strategies[0]][0].classes
    print(f"{len(classes)} of {y_true.shape[1]} classes have positives and negatives in PTB-XL")

    # Point estimates (full test set) and SD across seeds.
    ones = np.ones(n)
    point = {s: [r.compute(ones) for r in runs[s]] for s in args.strategies}

    # Paired bootstrap, one process per strategy if --n_jobs > 1.
    print(f"bootstrap: {args.n_boot} resamples x {len(args.strategies)} strategies x {len(args.seeds)} seeds", flush=True)
    if args.n_jobs > 1:
        with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
            futures = {s: ex.submit(bootstrap_strategy, runs[s], n, args.n_boot, args.boot_seed)
                       for s in args.strategies}
            per_seed = {s: f.result() for s, f in futures.items()}
    else:
        per_seed = {s: bootstrap_strategy(runs[s], n, args.n_boot, args.boot_seed) for s in args.strategies}
    ids = list(STRATEGIES)
    boot = {s: resample_seeds(per_seed[s], args.boot_seed, ids.index(s)) for s in args.strategies}
    boot_test = {s: {m: v.mean(axis=1) for m, v in per_seed[s].items()} for s in args.strategies}
    if len(args.seeds) < 3:
        print(f"WARNING: only {len(args.seeds)} seed(s); the two-level bootstrap cannot estimate "
              "training variability reliably.", flush=True)

    rows = []
    for s in args.strategies:
        for m in METRICS:
            seed_vals = [p[m] for p in point[s]]
            lo, hi = np.nanpercentile(boot[s][m], [2.5, 97.5])
            lo_t, hi_t = np.nanpercentile(boot_test[s][m], [2.5, 97.5])
            rows.append({"strategy": s, "metric": m, "estimate": np.mean(seed_vals),
                         "ci_low": lo, "ci_high": hi, "ci_low_test": lo_t, "ci_high_test": hi_t,
                         "sd_seeds": np.std(seed_vals, ddof=1) if len(seed_vals) > 1 else np.nan,
                         "per_seed": ";".join(f"{v:.4f}" for v in seed_vals)})
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(args.out_folder, "results_summary.csv"), index=False)

    # Main table, ordered by micro-AUROC.
    est = summary.pivot(index="strategy", columns="metric", values="estimate")
    order = est["micro_auroc"].sort_values(ascending=False).index.tolist()
    with open(os.path.join(args.out_folder, "results_main_table.tex"), "w") as f:
        for s in order:
            cells = [STRATEGIES[s]]
            for m in ["micro_auroc", "sd", "macro_auroc", "micro_auprc", "macro_auprc",
                      "macro_f1", "macro_sens", "macro_spec"]:
                if m == "sd":
                    sd = summary.query("strategy == @s and metric == 'micro_auroc'")["sd_seeds"].iloc[0]
                    cells.append(f"{sd:.4f}")
                    continue
                r = summary.query("strategy == @s and metric == @m").iloc[0]
                cells.append(fmt(r.estimate, r.ci_low, r.ci_high))
            f.write(" & ".join(cells) + r" \\" + "\n")

    # All pairwise differences in micro- and macro-AUROC, Holm-adjusted within each
    # metric. Primary: two-level bootstrap; *_test: recording-only bootstrap.
    all_pairs = []
    for metric in ["micro_auroc", "macro_auroc"]:
        rows_m = []
        for i, a in enumerate(order):
            for b_ in order[i + 1:]:
                d = boot[b_][metric] - boot[a][metric]
                d_t = boot_test[b_][metric] - boot_test[a][metric]
                lo, hi = np.percentile(d, [2.5, 97.5])
                lo_t, hi_t = np.percentile(d_t, [2.5, 97.5])
                rows_m.append({"metric": metric, "strategy": b_, "comparator": a,
                               "delta": est.loc[b_, metric] - est.loc[a, metric],
                               "ci_low": lo, "ci_high": hi, "p": boot_p(d),
                               "ci_low_test": lo_t, "ci_high_test": hi_t, "p_test": boot_p(d_t)})
        rows_m = pd.DataFrame(rows_m)
        rows_m["p_holm"] = holm(rows_m["p"].values)
        rows_m["p_holm_test"] = holm(rows_m["p_test"].values)
        all_pairs.append(rows_m)
    all_pairs = pd.concat(all_pairs, ignore_index=True)
    all_pairs.to_csv(os.path.join(args.out_folder, "results_pairwise.csv"), index=False)
    with open(os.path.join(args.out_folder, "results_pairwise_macro_table.tex"), "w") as f:
        for r in all_pairs.query("metric == 'macro_auroc'").itertuples():
            f.write(f"{STRATEGIES[r.strategy]} & {STRATEGIES[r.comparator]} & "
                    f"{fmt(r.delta, r.ci_low, r.ci_high)} & {fmt_p(r.p_holm, args.n_boot)} \\\\\n")
    pairs = all_pairs.query("metric == 'micro_auroc'")
    with open(os.path.join(args.out_folder, "results_pairwise_table.tex"), "w") as f:
        for r in pairs.itertuples():
            f.write(f"{STRATEGIES[r.strategy]} & {STRATEGIES[r.comparator]} & "
                    f"{fmt(r.delta, r.ci_low, r.ci_high)} & {fmt_p(r.p_holm, args.n_boot)} \\\\\n")

    # Paired DeLong tests (secondary; per seed).
    if not args.no_delong:
        print("DeLong tests", flush=True)
        run_delong(y_true, preds, order, classes, args.seeds, args.out_folder)

    # Per-class AUROC / AUPRC (point estimates, mean over seeds).
    names = [f"class_{j}" for j in range(y_true.shape[1])]
    if args.class_names:
        mapping = pd.read_csv(args.class_names)
        codes = (pd.read_csv(args.classes)["SNOMEDCTCode"].astype(str).tolist() if args.classes
                 else sorted(mapping["SNOMEDCTCode"].astype(str)))
        abbr = dict(zip(mapping["SNOMEDCTCode"].astype(str), mapping["Abbreviation"]))
        names = [abbr[c] for c in codes]
    per_class = pd.DataFrame({"class": [names[j] for j in classes],
                              "n_pos": y_true[:, classes].sum(0).astype(int)})
    for s in args.strategies:
        per_class[f"{s}_auroc"] = np.mean([p["per_class_auroc"] for p in point[s]], axis=0)
        per_class[f"{s}_auprc"] = np.mean([p["per_class_auprc"] for p in point[s]], axis=0)
    per_class.to_csv(os.path.join(args.out_folder, "results_per_class.csv"), index=False)

    # Optional subgroup point estimates (by sex and age group).
    meta = pd.read_csv(args.meta) if args.meta else None
    if meta is not None and len(meta) != n:
        print(f"Subgroups skipped: {args.meta} has {len(meta)} rows, predictions have {n} (smoke test?)")
        meta = None
    if meta is not None:
        groups = {f"sex={v}": (meta["sex"] == v).values for v in meta["sex"].dropna().unique()}
        age = meta["age"]
        for lo_age, hi_age in [(0, 40), (40, 60), (60, 80), (80, 200)]:
            groups[f"age {lo_age}-{hi_age - 1}"] = ((age >= lo_age) & (age < hi_age)).values
        sub = []
        for g, mask in groups.items():
            w = mask.astype(np.float64)
            for s in args.strategies:
                vals = [r.compute(w) for r in runs[s]]
                sub.append({"group": g, "n": int(mask.sum()), "strategy": s,
                            "micro_auroc": np.mean([v["micro_auroc"] for v in vals]),
                            "macro_auroc": np.mean([v["macro_auroc"] for v in vals])})
        pd.DataFrame(sub).to_csv(os.path.join(args.out_folder, "results_subgroups.csv"), index=False)

    print(summary.query("metric == 'micro_auroc'").to_string(index=False))


if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))
