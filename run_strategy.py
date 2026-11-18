#!/usr/bin/env python

# Re-runs one training strategy for the revision and saves the predictions needed
# for confidence intervals and statistical tests. Run from the repository root:
#
#   python run_strategy.py -d data -s fedavg --seed 1 -o results_revision
#
# Strategies: cds, iil, ciil, ciil_1epoch, fedavg, fedavg_repeated, fedsgd
# Normally started by run_all.py. See REVISION.md.
#
# Differences from the original scripts:
#   * FedAvg / repeated FedAvg: every centre trains its OWN copy of the model,
#     starting from the same global weights (the original shared one model object,
#     so "local" training was sequential). INCART is included. Aggregation is
#     weighted by local training set size.
#   * FedSGD: correct loss (from_logits=False), weighted gradient average,
#     early stopping on mean validation AUROC instead of a fixed 100/200 rounds.
#   * CIIL: the mean validation AUROC is computed for the model at the END of each
#     cycle, and the reported model is the one selected by validation.
#   * PTB-XL is evaluated once, after training. It is never used during training.
#
# Checkpoints: the run can be stopped at any time (Ctrl+C, shutdown) and resumes
# from <out>/<strategy>_seed<k>/checkpoint.pkl when started again. A checkpoint is
# written after each centre (IIL, FedAvg), each cycle (CIIL) and each evaluation
# (FedSGD, every 200 rounds). It holds model weights, optimizer (Adam) state, the
# best model so far, the history and the random-number generator states. Work
# since the last checkpoint (at most one centre's training) is repeated. CDS is a
# single training run (~25 min) and restarts from scratch. Use --fresh to ignore
# an existing checkpoint.

import os

os.environ.setdefault("KERAS_BACKEND", "torch")  # GPU on native Windows via PyTorch

import argparse
import json
import pickle
import random
import sys
import time

import keras
import numpy as np
import pandas as pd

from src.centres import centre as centre_module
from src.centres.centre import (CentralModelDistributor, CentralTrainer, ExternalValidationHospital,
                                LocalHospital)
from src.models.model import build_iception_model

INPUT_SHAPE = (1000, 12)
N_CLASSES = 30

# Fixed visiting order used in all strategies (as in the original scripts).
CENTRES = [
    ("st_petersburg", "stpeter"),
    ("ptb_diag", "ptbdiag"),
    ("chapman", "chapman"),
    ("ningbo", "ningbo"),
    ("georgia", "georgia"),
    ("china", "chinaphys"),
]
STRATEGIES = ["cds", "iil", "ciil", "ciil_1epoch", "fedavg", "fedavg_repeated", "fedsgd"]


def get_parser():
    parser = argparse.ArgumentParser(description="Re-run one training strategy and save predictions.")
    parser.add_argument("-d", "--data_folder", type=str, required=True)
    parser.add_argument("-s", "--strategy", type=str, required=True, choices=STRATEGIES)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("-o", "--out_folder", type=str, default="results_revision")
    parser.add_argument("--ciil_cycles", type=int, default=10)
    parser.add_argument("--ciil_1epoch_cycles", type=int, default=100)
    parser.add_argument("--fedavg_rounds", type=int, default=10, help="rounds of repeated FedAvg")
    parser.add_argument("--fedsgd_eval_every", type=int, default=200)
    parser.add_argument("--fedsgd_patience", type=int, default=5)
    parser.add_argument("--fedsgd_max_rounds", type=int, default=20000)
    parser.add_argument("--smoke", action="store_true",
                        help="tiny run (300 recordings per centre, 2 epochs) to test the pipeline")
    parser.add_argument("--fresh", action="store_true", help="ignore an existing checkpoint")
    return parser


# ----------------------------------------------------------------------------- checkpoints

def _rng_state():
    state = {"python": random.getstate(), "numpy": np.random.get_state()}
    if keras.backend.backend() == "torch":
        import torch
        state["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    if "torch" in state:
        import torch
        torch.set_rng_state(state["torch"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])


def optimizer_state(opt):
    return {"built": bool(opt.built), "vars": [np.array(keras.ops.convert_to_numpy(v)) for v in opt.variables]}


def set_optimizer_state(opt, train_vars, state):
    if not state["built"]:
        return
    if not opt.built:
        opt.build(train_vars)
    if len(opt.variables) != len(state["vars"]):
        raise RuntimeError("optimizer state does not match the model; delete checkpoint.pkl or use --fresh")
    for v, value in zip(opt.variables, state["vars"]):
        v.assign(value)


def model_state(model):
    return {"weights": model.get_weights(), "optimizer": optimizer_state(model.optimizer)}


def set_model_state(model, state):
    model.set_weights(state["weights"])
    set_optimizer_state(model.optimizer, model.trainable_variables, state["optimizer"])


class Checkpoint:
    def __init__(self, out_dir, enabled=True):
        self.path = os.path.join(out_dir, "checkpoint.pkl")
        self.state = None
        self.start = time.time()
        self.elapsed_before = 0.0
        if enabled and os.path.exists(self.path):
            with open(self.path, "rb") as f:
                self.state = pickle.load(f)
            self.elapsed_before = self.state["elapsed"]
            print(f"RESUMING from checkpoint: {self.state['desc']} "
                  f"({self.elapsed_before / 3600:.1f} h of training already done)", flush=True)

    def elapsed(self):
        return self.elapsed_before + time.time() - self.start

    def save(self, desc, models, **state):
        payload = dict(state, desc=desc, elapsed=self.elapsed(), rng=_rng_state(),
                       models={k: model_state(m) for k, m in models.items()})
        tmp = self.path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, self.path)  # atomic: an interrupted save never corrupts the checkpoint
        print(f"checkpoint saved: {desc}", flush=True)

    def restore(self, models):
        """Restore models and RNG states; returns the saved state dict."""
        for k, m in models.items():
            set_model_state(m, self.state["models"][k])
        _set_rng_state(self.state["rng"])
        return self.state

    def remove(self):
        if os.path.exists(self.path):
            os.remove(self.path)


# ----------------------------------------------------------------------------- helpers

def load_centres(data_folder, mode, subsample=None):
    """mode: 'shared' (one model for all; sequential strategies), 'own' (one model
    per centre; FedAvg) or 'none' (model attached later; CDS, FedSGD)."""
    shared = build_iception_model(INPUT_SHAPE, N_CLASSES) if mode == "shared" else None
    centres = []
    for name, file_key in CENTRES:
        c = LocalHospital(name, os.path.join(data_folder, f"X_data_{file_key}.npy"),
                          os.path.join(data_folder, f"y_data_{file_key}.npy"), subsample=subsample)
        if mode == "shared":
            c.load_model(shared)
        elif mode == "own":
            c.load_model(build_iception_model(INPUT_SHAPE, N_CLASSES))
        centres.append(c)
    return centres


def mean_val_auroc(centres):
    """Unweighted mean of the centres' validation AUROCs for each centre's current weights.
    A centre whose validation set has no positive label (AUROC undefined) is left
    out of the mean, with a warning."""
    rocs = [c.predict_val()[2] for c in centres]
    undefined = [c.name for c, r in zip(centres, rocs) if np.isnan(r)]
    if undefined:
        print(f"WARNING: validation AUROC undefined for {undefined} (no positive labels); left out of the mean")
    return float(np.nanmean(rocs)) if len(undefined) < len(rocs) else float("nan"), rocs


def is_better(value, best):
    return best is None or (not np.isnan(value) and (np.isnan(best) or value > best))


def set_all_weights(centres, weights):
    for c in centres:
        c.set_weights(weights)


def centre_names():
    return [n for n, _ in CENTRES]


# ----------------------------------------------------------------------------- strategies

def run_cds(centres, history, ck):
    central_trainer = CentralTrainer(INPUT_SHAPE, N_CLASSES)
    for c in centres:
        central_trainer.load_train_data_from_centre(*c.get_data(split="train"))
        central_trainer.load_val_data_from_centre(*c.get_data(split="val"))
    central_trainer.train_to_convergence(os.path.join(history["dir"], "cds_history.csv"))
    model = central_trainer.get_model()
    for c in centres:
        c.load_model(model)
    _, rocs = mean_val_auroc(centres)
    history["rows"].append(dict(zip(centre_names(), rocs)))
    return model.get_weights()


def run_iil(centres, history, ck):
    models = {"shared": centres[0].model}
    start = 0
    if ck.state:
        start = ck.restore(models)["next_centre"]
    for i in range(start, len(centres)):
        c = centres[i]
        c.train_to_convergence(os.path.join(history["dir"], f"iil_{c.name}_history.csv"))
        ck.save(f"IIL, trained {c.name}", models, next_centre=i + 1)
    _, rocs = mean_val_auroc(centres)
    history["rows"].append(dict(zip(centre_names(), rocs)))
    return centres[-1].get_weights()


def run_ciil(centres, history, n_cycles, one_epoch, ck):
    models = {"shared": centres[0].model}
    best_mean_val, best_weights, start = None, None, 0
    if ck.state:
        s = ck.restore(models)
        start, best_mean_val, best_weights = s["next_cycle"], s["best_mean_val"], s["best_weights"]
        history["rows"], history["selected"] = s["rows"], s["selected"]
    for cycle in range(start, n_cycles):
        for c in centres:
            if one_epoch:
                c.train_one_epoch()
            else:
                c.train_to_convergence(os.path.join(history["dir"], f"ciil_{c.name}_cycle_{cycle}.csv"))
        # All centres share one model, so this evaluates the end-of-cycle model.
        temp_mean_val, rocs = mean_val_auroc(centres)
        history["rows"].append(dict(zip(centre_names(), rocs), cycle=cycle, mean_val=temp_mean_val))
        print(f"cycle {cycle}: mean val AUROC {temp_mean_val:.4f}", flush=True)
        if is_better(temp_mean_val, best_mean_val):
            history["selected"] = len(history["rows"]) - 1
            best_mean_val, best_weights = temp_mean_val, centres[-1].get_weights()
        ck.save(f"CIIL, finished cycle {cycle + 1}/{n_cycles}", models, next_cycle=cycle + 1,
                best_mean_val=best_mean_val, best_weights=best_weights,
                rows=history["rows"], selected=history.get("selected"))
    return best_weights


def run_fedavg(centres, history, n_rounds, ck):
    models = {f"centre{i}": c.model for i, c in enumerate(centres)}
    sizes = [c.n_train() for c in centres]
    global_weights = centres[0].get_weights()  # common initialisation
    best_mean_val, best_weights, start_round, start_centre, local_weights = None, None, 0, 0, []
    if ck.state:
        s = ck.restore(models)
        start_round, start_centre, local_weights = s["round"], s["next_centre"], s["local_weights"]
        global_weights, best_mean_val, best_weights = s["global_weights"], s["best_mean_val"], s["best_weights"]
        history["rows"], history["selected"] = s["rows"], s["selected"]

    def save(desc, rnd, next_centre):
        ck.save(desc, models, round=rnd, next_centre=next_centre, local_weights=local_weights,
                global_weights=global_weights, best_mean_val=best_mean_val, best_weights=best_weights,
                rows=history["rows"], selected=history.get("selected"))

    for rnd in range(start_round, n_rounds):
        first = start_centre if rnd == start_round else 0
        if first == 0:
            local_weights = []
        for i in range(first, len(centres)):
            c = centres[i]
            c.set_weights(global_weights)
            c.train_to_convergence(os.path.join(history["dir"], f"fedavg_{c.name}_round_{rnd}.csv"))
            local_weights.append(c.get_weights())
            if i < len(centres) - 1:
                save(f"FedAvg round {rnd + 1}/{n_rounds}, trained {c.name}", rnd, i + 1)
        global_weights = CentralModelDistributor.weighted_avg_weights(local_weights, sizes)
        set_all_weights(centres, global_weights)
        temp_mean_val, rocs = mean_val_auroc(centres)
        history["rows"].append(dict(zip(centre_names(), rocs), round=rnd, mean_val=temp_mean_val))
        print(f"round {rnd}: mean val AUROC {temp_mean_val:.4f}", flush=True)
        if is_better(temp_mean_val, best_mean_val):
            history["selected"] = len(history["rows"]) - 1
            best_mean_val, best_weights = temp_mean_val, global_weights
        local_weights = []
        save(f"FedAvg, finished round {rnd + 1}/{n_rounds}", rnd + 1, 0)
    return best_weights


def run_fedsgd(centres, history, eval_every, patience, max_rounds, ck):
    # A single shared model is correct here: all centres compute their gradient at
    # the same global weights, and only the server updates the weights.
    central_unit = CentralModelDistributor(INPUT_SHAPE, N_CLASSES)
    model = central_unit.get_model()
    for c in centres:
        c.load_model(model)
        c.set_generator()
    sizes = [c.n_train() for c in centres]
    best_mean_val, best_weights, evals_without_improvement, start = None, None, 0, 1
    if ck.state:
        s = ck.restore({"server": model})
        set_optimizer_state(central_unit.optimizer, model.trainable_variables, s["server_optimizer"])
        start, best_mean_val, best_weights = s["next_round"], s["best_mean_val"], s["best_weights"]
        evals_without_improvement = s["evals_without_improvement"]
        history["rows"], history["selected"] = s["rows"], s["selected"]
    for rnd in range(start, max_rounds + 1):
        for c in centres:
            central_unit.update_grads_list(c.train_one_batch())
        central_unit.apply_weighted_avg_grads(sizes)
        if rnd % eval_every == 0:
            temp_mean_val, rocs = mean_val_auroc(centres)
            history["rows"].append(dict(zip(centre_names(), rocs), round=rnd, mean_val=temp_mean_val))
            print(f"round {rnd}: mean val AUROC {temp_mean_val:.4f}", flush=True)
            if is_better(temp_mean_val, best_mean_val):
                history["selected"] = len(history["rows"]) - 1
                best_mean_val, best_weights = temp_mean_val, central_unit.get_weights_from_model()
                evals_without_improvement = 0
            else:
                evals_without_improvement += 1
            ck.save(f"FedSGD, round {rnd}", {"server": model}, next_round=rnd + 1,
                    server_optimizer=optimizer_state(central_unit.optimizer),
                    best_mean_val=best_mean_val, best_weights=best_weights,
                    evals_without_improvement=evals_without_improvement,
                    rows=history["rows"], selected=history.get("selected"))
            if evals_without_improvement >= patience:
                break
    if best_weights is None:  # never evaluated (max_rounds < eval_every)
        best_weights = central_unit.get_weights_from_model()
    return best_weights


# ----------------------------------------------------------------------------- main

def run(args):
    subsample = None
    if args.smoke:
        subsample = 300
        centre_module.TRAIN_CONFIG.update(max_epochs=2, patience=1)
        args.ciil_cycles, args.ciil_1epoch_cycles, args.fedavg_rounds = 2, 2, 2
        args.fedsgd_eval_every, args.fedsgd_max_rounds = 5, 10

    keras.utils.set_random_seed(args.seed)
    run_name = f"{args.strategy}_seed{args.seed}"
    out_dir = os.path.join(args.out_folder, run_name)
    os.makedirs(out_dir, exist_ok=True)
    history = {"dir": out_dir, "rows": []}
    print(f"{run_name}: Keras {keras.__version__}, backend {keras.backend.backend()}", flush=True)

    mode = {"iil": "shared", "ciil": "shared", "ciil_1epoch": "shared",
            "fedavg": "own", "fedavg_repeated": "own"}.get(args.strategy, "none")
    centres = load_centres(args.data_folder, mode, subsample)
    if mode == "own":
        set_all_weights(centres, centres[0].get_weights())  # same initial weights everywhere
    ck = Checkpoint(out_dir, enabled=not args.fresh)

    if args.strategy == "cds":
        final_weights = run_cds(centres, history, ck)
    elif args.strategy == "iil":
        final_weights = run_iil(centres, history, ck)
    elif args.strategy == "ciil":
        final_weights = run_ciil(centres, history, args.ciil_cycles, False, ck)
    elif args.strategy == "ciil_1epoch":
        final_weights = run_ciil(centres, history, args.ciil_1epoch_cycles, True, ck)
    elif args.strategy == "fedavg":
        final_weights = run_fedavg(centres, history, 1, ck)
    elif args.strategy == "fedavg_repeated":
        final_weights = run_fedavg(centres, history, args.fedavg_rounds, ck)
    elif args.strategy == "fedsgd":
        final_weights = run_fedsgd(centres, history, args.fedsgd_eval_every,
                                   args.fedsgd_patience, args.fedsgd_max_rounds, ck)
    elapsed = ck.elapsed()

    # Validation predictions of the final model, pooled over centres (for thresholds).
    set_all_weights(centres, final_weights)
    val_true, val_pred = zip(*[c.predict_val_raw() for c in centres])
    np.save(os.path.join(out_dir, "val_true.npy"), np.vstack(val_true))
    np.save(os.path.join(out_dir, "val_pred.npy"), np.vstack(val_pred))
    np.save(os.path.join(out_dir, "val_centre.npy"),
            np.concatenate([np.full(len(v), i) for i, v in enumerate(val_true)]))
    model = centres[0].model
    del centres

    # External test: evaluated once, after training.
    ptb_xl = ExternalValidationHospital("PTB-XL", os.path.join(args.data_folder, "X_data_ptbxl.npy"),
                                        os.path.join(args.data_folder, "y_data_ptbxl.npy"), subsample=subsample)
    ptb_xl.load_model(model)
    ptb_xl.set_weights(final_weights)
    pred = ptb_xl.model.predict(ptb_xl.X_data, verbose=0)
    np.save(os.path.join(out_dir, "ptbxl_pred.npy"), pred)
    np.save(os.path.join(args.out_folder, "ptbxl_true.npy"), ptb_xl.y_data)
    ptb_xl.model.save_weights(os.path.join(out_dir, "final.weights.h5"))
    _, _, auroc = ptb_xl.predict()

    pd.DataFrame(history["rows"]).to_csv(os.path.join(out_dir, "training_history.csv"), index=False)
    # run_info.json is written last: run_all.py treats its presence as "run finished".
    with open(os.path.join(out_dir, "run_info.json"), "w") as f:
        json.dump({"strategy": args.strategy, "seed": args.seed, "train_time_s": elapsed,
                   "selected_round": history.get("selected"),
                   "resumed_from_checkpoint": ck.state is not None,
                   "ptbxl_micro_auroc": float(auroc),
                   "n_params": int(ptb_xl.model.count_params()),
                   "keras": keras.__version__, "backend": keras.backend.backend(),
                   "python": sys.version.split()[0], "smoke": bool(args.smoke)}, f, indent=2)
    ck.remove()
    print(f"{run_name}: done in {elapsed:.0f} s, PTB-XL micro-AUROC {auroc:.4f}", flush=True)


if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))
