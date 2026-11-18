#!/usr/bin/env python

# Builds Supplementary Tables S1 (diagnosis counts per centre) and S2 (age/sex per
# centre), and ptbxl_meta.csv for the subgroup analysis, from the files written by
# prepare_data.py:
#
#   python dataset_stats.py -d data -m raw/.../dx_mapping_scored.csv -o tables
#
# Label counts come from y_data_*.npy, i.e. exactly what the models saw.

import argparse
import os
import sys

import numpy as np
import pandas as pd

# (column label, npy key)
CENTRES = [("INC", "stpeter"), ("PTB", "ptbdiag"), ("CHA", "chapman"), ("NIN", "ningbo"),
           ("GEO", "georgia"), ("CPS", "chinaphys"), ("XL", "ptbxl")]


def get_parser():
    parser = argparse.ArgumentParser(description="Dataset tables for the supplement.")
    parser.add_argument("-d", "--data_folder", type=str, required=True, help="folder written by prepare_data.py")
    parser.add_argument("-m", "--mapping", type=str, required=True, help="dx_mapping_scored.csv")
    parser.add_argument("-o", "--out_folder", type=str, default="tables")
    return parser


def run(args):
    os.makedirs(args.out_folder, exist_ok=True)
    mapping = pd.read_csv(args.mapping)
    mapping["code"] = mapping["SNOMEDCTCode"].astype(str)
    mapping = mapping.drop_duplicates("code").set_index("code")
    classes_file = os.path.join(args.data_folder, "classes.csv")
    codes = (pd.read_csv(classes_file)["SNOMEDCTCode"].astype(str).tolist() if os.path.exists(classes_file)
             else sorted(mapping.index))

    counts, totals, unlabeled = {}, {}, {}
    for label, key in CENTRES:
        y = np.load(os.path.join(args.data_folder, f"y_data_{key}.npy"))
        assert y.shape[1] == len(codes), f"{key}: {y.shape[1]} columns, {len(codes)} codes"
        counts[label] = y.sum(axis=0).astype(int)
        totals[label] = y.shape[0]
        unlabeled[label] = int((y.sum(axis=1) == 0).sum())

    table = pd.DataFrame({"Diagnosis": mapping.loc[codes, "Dx"].values,
                          "Abbreviation": mapping.loc[codes, "Abbreviation"].values,
                          "SNOMED CT": codes})
    for label, _ in CENTRES:
        table[label] = counts[label]
    table = table.sort_values("Diagnosis", key=lambda s: s.str.lower())
    table.to_csv(os.path.join(args.out_folder, "table_S1_diagnoses.csv"), index=False)
    with open(os.path.join(args.out_folder, "table_S1_diagnoses.tex"), "w") as f:
        for r in table.itertuples(index=False):
            cells = [str(r[0]), str(r[1]), str(r[2])] + [f"{int(v):,}" for v in r[3:]]
            f.write(" & ".join(cells) + r" \\" + "\n")
        f.write(r"\midrule" + "\n")
        f.write("No scored diagnosis & & & " + " & ".join(f"{unlabeled[c]:,}" for c, _ in CENTRES) + r" \\" + "\n")
        f.write("Total recordings & & & " + " & ".join(f"{totals[c]:,}" for c, _ in CENTRES) + r" \\" + "\n")
    print(f"PTB-XL: {int((counts['XL'] > 0).sum())} of {len(codes)} diagnoses have at least one positive")

    rows = []
    for label, key in CENTRES:
        meta_file = os.path.join(args.data_folder, f"meta_{key}.csv")
        if not os.path.exists(meta_file):
            print(f"{meta_file} missing; skipping Table S2")
            return
        demo = pd.read_csv(meta_file)
        assert len(demo) == totals[label], f"{key}: {len(demo)} meta rows vs {totals[label]} recordings"
        if key == "ptbxl":
            demo[["age", "sex"]].to_csv(os.path.join(args.out_folder, "ptbxl_meta.csv"), index=False)
        q1, med, q3 = demo["age"].quantile([0.25, 0.5, 0.75])
        rows.append({"centre": label, "n": len(demo), "age_median": med, "age_q1": q1, "age_q3": q3,
                     "female_pct": 100 * (demo["sex"] == "Female").sum() / demo["sex"].notna().sum(),
                     "missing_age": int(demo["age"].isna().sum()),
                     "missing_sex": int(demo["sex"].isna().sum())})
    demo_table = pd.DataFrame(rows)
    demo_table.to_csv(os.path.join(args.out_folder, "table_S2_demographics.csv"), index=False)
    print(demo_table.to_string(index=False))


if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))
