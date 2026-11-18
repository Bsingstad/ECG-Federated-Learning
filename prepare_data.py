#!/usr/bin/env python

# Builds the .npy files used for training from the raw PhysioNet/CinC Challenge
# 2021 WFDB folders, with the same steps as notebooks/Structure_and_label_data.ipynb:
#   * labels: the scored SNOMED CT codes (dx_mapping_scored.csv) that occur in the
#     six training databases, sorted as strings, one-hot encoded;
#   * signals: zero-padded / truncated (post) to 10 s at the original sampling
#     rate if the length is not 10 s (integer seconds), then resampled to 100 Hz
#     (1000 samples) with scipy.signal.resample; raw ADC units, no normalisation;
#   * CPSC = CPSC 2018 + CPSC 2018 extra (the notebook merged Training_2 into
#     Training_WFDB); recordings are processed in sorted file-name order.
# It also writes meta_<key>.csv (record, age, sex) for every database.
#
#   python prepare_data.py --raw_root raw --out data            # raw data already unpacked
#   python prepare_data.py --raw_root raw --out data --download # download from Kaggle first
#
# --download uses the Kaggle mirrors used for the original study and needs
# `pip install kaggle` and an API token (https://www.kaggle.com/settings ->
# "Create New Token", saved as %USERPROFILE%\.kaggle\kaggle.json).

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import loadmat

KAGGLE_DATASETS = [
    "physionet/georgia-12lead-ecg-challenge-database",
    "physionet/st-petersburg-incart-12lead-arrhythmia-database",
    "physionet/ptbxl-electrocardiography-database",
    "physionet/ptb-diagnostic-ecg-database",
    "physionet/china-12lead-ecg-challenge-database",
    "physionet/china-physiological-signal-challenge-in-2018",
    "bjoernjostein/physionet-challenge-2021-snomed-mappings",
    "erarayamorenzomuten/chapmanshaoxing-12lead-ecg-database",
    "erarayamorenzomuten/ningbo-first-hospital-12lead-ecg-database",
]

# npy key -> alternative sets of folder names (Kaggle mirror names first, then the
# names in the PhysioNet challenge-2021 tarballs and in the PhysioNet 1.0.3 tree).
SOURCES = {
    "stpeter": [["Training_StPetersburg"], ["WFDB_StPetersburg"], ["st_petersburg_incart"]],
    "ptbdiag": [["Training_PTB"], ["WFDB_PTB"], ["ptb"]],
    "chapman": [["WFDB_ChapmanShaoxing"], ["chapman_shaoxing"]],
    "ningbo": [["WFDB_Ningbo"], ["ningbo"]],
    "georgia": [["Georgia"], ["WFDB_Ga"], ["georgia"]],
    "chinaphys": [["Training_WFDB", "Training_2"], ["WFDB_CPSC2018", "WFDB_CPSC2018_2"],
                  ["cpsc_2018", "cpsc_2018_extra"]],
    "ptbxl": [["WFDB"], ["WFDB_PTBXL"], ["ptb-xl"]],
}
TRAINING_KEYS = ["stpeter", "ptbdiag", "chapman", "ningbo", "georgia", "chinaphys"]


def get_parser():
    parser = argparse.ArgumentParser(description="Raw WFDB challenge data -> .npy")
    parser.add_argument("--raw_root", type=str, default="raw")
    parser.add_argument("--out", type=str, default="data")
    parser.add_argument("--mapping", type=str, default=None,
                        help="dx_mapping_scored.csv (default: searched for under --raw_root)")
    parser.add_argument("--download", action="store_true", help="download the Kaggle mirrors first")
    parser.add_argument("--n_jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    return parser


def download(raw_root):
    os.makedirs(raw_root, exist_ok=True)
    for ds in KAGGLE_DATASETS:
        print(f"downloading {ds}", flush=True)
        subprocess.run(["kaggle", "datasets", "download", "-d", ds, "-p", raw_root, "--unzip"], check=True)


def find_dirs(raw_root):
    by_name = {}
    for root, dirs, _ in os.walk(raw_root):
        for d in dirs:
            by_name.setdefault(d, []).append(os.path.join(root, d))
    # Zip files often unpack to nested folders with the same name (WFDB/WFDB): keep
    # only the outermost one (records are collected recursively below it).
    for name, paths in by_name.items():
        paths = [os.path.normpath(p) for p in paths]
        by_name[name] = [p for p in paths if not any(p != q and p.startswith(q + os.sep) for q in paths)]
    found = {}
    for key, alternatives in SOURCES.items():
        for names in alternatives:
            if all(n in by_name for n in names):
                for n in names:
                    if len(by_name[n]) > 1:
                        sys.exit(f"{key}: several folders named {n}: {by_name[n]}")
                found[key] = [by_name[n][0] for n in names]
                break
        else:
            sys.exit(f"{key}: none of {alternatives} found under {raw_root}")
    return found


def list_records(folders):
    """All .mat files below the folders, sorted by file name (as the notebook did)."""
    files = []
    for folder in folders:
        for root, _, names in os.walk(folder):
            files += [os.path.join(root, n) for n in names if n.endswith(".mat")]
    return sorted(files, key=os.path.basename)


def read_header(mat_file):
    with open(mat_file[:-4] + ".hea") as f:
        return f.readlines()


def get_dx(header):
    for line in header:
        if line.startswith("#Dx"):
            try:
                return [e.strip() for e in line.split(": ")[1].split(",")]
            except IndexError:
                return []
    return []


def get_demographics(header):
    age, sex = np.nan, None
    for line in header:
        if line.startswith("#Age"):
            try:
                age = float(line.split(":")[1])
            except ValueError:
                pass
        elif line.startswith("#Sex"):
            sex = {"male": "Male", "m": "Male", "female": "Female", "f": "Female"}.get(
                line.split(":")[1].strip().lower())
    return age, sex


def pad_and_resample(data, header, new_fs=100):
    """Same as pad_and_resample() in the notebook. keras pad_sequences(padding='post',
    truncating='post') with its default int32 dtype is reproduced with numpy."""
    current_fs = int(header[0].split(" ")[2])
    current_len = int(header[0].split(" ")[3])
    if current_len // current_fs != 10:
        maxlen = int(current_fs * 10)
        padded = np.zeros((data.shape[0], maxlen), dtype=np.int32)
        n = min(data.shape[1], maxlen)
        padded[:, :n] = data[:, :n].astype(np.int32)
    else:
        padded = data
    if current_fs != new_fs:
        return signal.resample(padded, 10 * new_fs, axis=1)
    return padded


def process_record(args):
    mat_file, scored = args
    data = np.asarray(loadmat(mat_file)["val"], dtype=np.float64)
    header = read_header(mat_file)
    x = pad_and_resample(data, header).astype(np.float32)
    dx = set(get_dx(header))
    y = np.array([1.0 if c in dx else 0.0 for c in scored], dtype=np.float32)
    age, sex = get_demographics(header)
    return x, y, os.path.basename(mat_file)[:-4], age, sex


def run(args):
    if args.download:
        download(args.raw_root)
    mapping_path = args.mapping
    if mapping_path is None:
        hits = [os.path.join(r, f) for r, _, fs in os.walk(args.raw_root) for f in fs if f == "dx_mapping_scored.csv"]
        if not hits:
            sys.exit("dx_mapping_scored.csv not found; pass --mapping")
        mapping_path = hits[0]
    scored_all = set(pd.read_csv(mapping_path)["SNOMEDCTCode"].astype(str))

    dirs = find_dirs(args.raw_root)
    records = {key: list_records(folders) for key, folders in dirs.items()}
    for key in dirs:
        print(f"{key}: {len(records[key])} recordings in {dirs[key]}")

    # Scored classes = scored codes present in the six training databases (as in the notebook).
    present = set()
    for key in TRAINING_KEYS:
        for f in records[key]:
            present |= set(get_dx(read_header(f)))
    scored = sorted(c for c in present if c in scored_all)
    print(f"{len(scored)} scored classes")
    if len(scored) != 30:
        print("WARNING: the study used 30 classes", file=sys.stderr)

    os.makedirs(args.out, exist_ok=True)
    pd.DataFrame({"SNOMEDCTCode": scored}).to_csv(os.path.join(args.out, "classes.csv"), index=False)
    with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
        for key, files in records.items():
            out = list(ex.map(process_record, [(f, scored) for f in files], chunksize=64))
            X = np.stack([o[0] for o in out])
            y = np.stack([o[1] for o in out])
            np.save(os.path.join(args.out, f"X_data_{key}.npy"), X)
            np.save(os.path.join(args.out, f"y_data_{key}.npy"), y)
            pd.DataFrame({"record": [o[2] for o in out], "age": [o[3] for o in out],
                          "sex": [o[4] for o in out]}).to_csv(os.path.join(args.out, f"meta_{key}.csv"), index=False)
            print(f"{key}: X {X.shape}, y {y.shape}, {int(y.sum())} labels", flush=True)
            del out, X, y


if __name__ == "__main__":
    run(get_parser().parse_args(sys.argv[1:]))
