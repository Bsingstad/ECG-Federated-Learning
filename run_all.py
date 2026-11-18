#!/usr/bin/env python

# One command for the whole revision pipeline (see REVISION.md):
#
#   python run_all.py --smoke          # ~15-30 min end-to-end test on a data subset
#   python run_all.py                  # full re-run: 7 strategies x 5 seeds + analysis
#   python run_all.py --seeds 1 2 3    # fewer seeds
#   python run_all.py --eval_only      # only the tables/statistics/figure
#
# Stop and start at any time (Ctrl+C, shutdown): finished runs (run_info.json)
# are skipped, and an interrupted run resumes from its last checkpoint (after each
# centre / cycle / round / FedSGD evaluation; see run_strategy.py). Each run is a
# separate process and logs to <results>/logs/<strategy>_seed<k>.log.

import argparse
import glob
import json
import os
import subprocess
import sys
import time

STRATEGIES = ["cds", "iil", "fedavg", "ciil_1epoch", "fedsgd", "ciil", "fedavg_repeated"]  # fast first


def get_parser():
    parser = argparse.ArgumentParser(description="Run all strategies and seeds, then the analysis.")
    parser.add_argument("-d", "--data_folder", default="data")
    parser.add_argument("-r", "--results_folder", default=None,
                        help="default: results_revision (results_smoke with --smoke)")
    parser.add_argument("-t", "--tables_folder", default=None,
                        help="default: <results_folder>/tables")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="default: 1 2 3 4 5 (only 1 with --smoke)")
    parser.add_argument("--strategies", nargs="+", default=STRATEGIES, choices=STRATEGIES)
    parser.add_argument("--mapping", default=None, help="dx_mapping_scored.csv (default: searched under raw/)")
    parser.add_argument("--smoke", action="store_true", help="tiny end-to-end test of the whole pipeline")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--n_jobs", type=int, default=4, help="processes for the bootstrap")
    return parser


def find_mapping(args):
    if args.mapping:
        return args.mapping
    hits = glob.glob(os.path.join("raw", "**", "dx_mapping_scored.csv"), recursive=True)
    return hits[0] if hits else None


def done(results, strategy, seed):
    return os.path.exists(os.path.join(results, f"{strategy}_seed{seed}", "run_info.json"))


def train_all(args, results):
    log_dir = os.path.join(results, "logs")
    os.makedirs(log_dir, exist_ok=True)
    todo = [(seed, s) for seed in args.seeds for s in args.strategies if not done(results, s, seed)]
    print(f"{len(todo)} runs to do ({len(args.seeds) * len(args.strategies) - len(todo)} already finished)")
    failed = []
    for i, (seed, strategy) in enumerate(todo, 1):
        name = f"{strategy}_seed{seed}"
        log = os.path.join(log_dir, f"{name}.log")
        cmd = [sys.executable, "-u", "run_strategy.py", "-d", args.data_folder, "-s", strategy,
               "--seed", str(seed), "-o", results] + (["--smoke"] if args.smoke else [])
        resuming = os.path.exists(os.path.join(results, name, "checkpoint.pkl"))
        print(f"[{i}/{len(todo)}] {time.strftime('%d.%m %H:%M')} {name}"
              f"{' (resuming from checkpoint)' if resuming else ''} (log: {log})", flush=True)
        start = time.time()
        with open(log, "a" if resuming else "w") as f:
            code = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
        if code != 0 or not done(results, strategy, seed):
            print(f"    FAILED (exit code {code}); see {log}", flush=True)
            failed.append(name)
            continue
        with open(os.path.join(results, name, "run_info.json")) as f:
            info = json.load(f)
        print(f"    done in {(time.time() - start) / 60:.0f} min, PTB-XL micro-AUROC "
              f"{info['ptbxl_micro_auroc']:.4f}", flush=True)
    return failed


def analyse(args, results, tables):
    os.makedirs(tables, exist_ok=True)
    py = sys.executable
    mapping = find_mapping(args)
    if mapping:
        subprocess.call([py, "dataset_stats.py", "-d", args.data_folder, "-m", mapping, "-o", tables])
    else:
        print("dx_mapping_scored.csv not found: skipping Tables S1-S2 and class names")

    seeds = [k for k in args.seeds if all(done(results, s, k) for s in args.strategies)]
    if not seeds:
        print("No seed is finished for all strategies yet; nothing to evaluate.")
        return
    print(f"Evaluating seeds {seeds}", flush=True)
    cmd = [py, "evaluate_with_ci.py", "-r", results, "-o", tables, "--seeds", *map(str, seeds),
           "--strategies", *args.strategies, "--n_boot", str(args.n_boot), "--n_jobs", str(args.n_jobs)]
    meta = os.path.join(tables, "ptbxl_meta.csv")
    if os.path.exists(meta):
        cmd += ["--meta", meta]
    if mapping:
        cmd += ["--class_names", mapping]
        classes = os.path.join(args.data_folder, "classes.csv")
        if os.path.exists(classes):
            cmd += ["--classes", classes]
    subprocess.call(cmd)
    subprocess.call([py, "plot_training_curves.py", "-r", results, "--seed", str(seeds[0]),
                     "-o", os.path.join(tables, "figS1_training_curves.pdf")])
    print(f"Tables and figures are in {tables}")


def run(args):
    results = args.results_folder or ("results_smoke" if args.smoke else "results_revision")
    tables = args.tables_folder or os.path.join(results, "tables")
    if args.seeds is None:
        args.seeds = [1] if args.smoke else [1, 2, 3, 4, 5]
    if args.smoke:
        args.n_boot = min(args.n_boot, 50)
    missing = [k for k in ["stpeter", "ptbdiag", "chapman", "ningbo", "georgia", "chinaphys", "ptbxl"]
               if not os.path.exists(os.path.join(args.data_folder, f"X_data_{k}.npy"))]
    if missing:
        sys.exit(f"Missing data for {missing} in {args.data_folder}. Run prepare_data.py first (see REVISION.md).")
    failed = [] if args.eval_only else train_all(args, results)
    analyse(args, results, tables)
    if failed:
        print(f"Failed runs: {failed}. Re-run the same command to retry them.")


if __name__ == "__main__":
    run(get_parser().parse_args(sys.argv[1:]))
