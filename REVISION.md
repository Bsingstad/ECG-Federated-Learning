# Revision re-run: how to produce the results requested by the reviewers

This branch (`revision`) re-trains all seven strategies with corrected code, saves
per-recording predictions, and computes everything the reviewers asked for:

| Reviewer request | Output (in `results_revision/tables/`) |
|---|---|
| More metrics with 95% CIs (R1 Major 5) | `results_main_table.tex`, `results_summary.csv` (micro/macro AUROC, AUPRC, F1, sensitivity, specificity, Brier) |
| Statistical comparisons (R1 Major 6, R2 W2) | `results_pairwise.csv` / `results_pairwise_table.tex` (micro-AUROC) and `results_pairwise_macro_table.tex` (macro-AUROC): two-level bootstrap, Holm; `results_delong_*.csv` / `results_delong_table.tex` (paired DeLong per seed, Holm; secondary) |
| Repeated-run variability (R2 W2) | included in every CI and test (two-level bootstrap); `sd_seeds` and `per_seed` columns in `results_summary.csv` |

**Statistics.** The primary CIs and tests use a two-level bootstrap: each
resample draws PTB-XL recordings (shared by all strategies, so comparisons are
paired on the test set) and, independently for each strategy, training seeds with
replacement. They therefore include run-to-run training variability, not only
test-set sampling. The recording-only bootstrap (training runs held fixed) is kept
in the `*_test` columns as a sensitivity analysis; with seeds 1–2 its CIs were
5–10 times narrower than the seed-to-seed spread. DeLong tests hold the trained
model fixed in the same way and are reported per seed as a secondary analysis.
| Per-diagnosis results | `results_per_class.csv` |
| Dataset tables (R1 Major 4) | `table_S1_diagnoses.tex` (30 diagnoses × centre), `table_S2_demographics.csv` (age/sex) |
| Subgroups (TRIPOD+AI 14, 23a) | `results_subgroups.csv` (by sex and age group) |
| Fig. 4 (ROC, legend sorted by AUROC) | `roc_<strategy>_dwnsmp.csv` |
| Training curves | `figS1_training_curves.pdf` |
| Training time (Table S4) | `results_revision/<strategy>_seed<k>/run_info.json` → `train_time_s` |

## Setup (Windows, Anaconda env `gpu1`)

TensorFlow ≥ 2.11 cannot use the GPU on native Windows, so the code now uses
Keras 3 with the **PyTorch backend** (`KERAS_BACKEND=torch`, set automatically by
`run_strategy.py`). `gpu1` already has everything needed (Keras 3.15, PyTorch 2.11
+ CUDA 12.8, scikit-learn, SciPy, pandas, matplotlib, h5py). Only the optional
Kaggle download needs an extra package.

```bat
conda activate gpu1
cd C:\Users\bjsin\Documents\GitHub\ECG-Federated-Learning
python -c "import torch; print(torch.cuda.is_available())"   :: must print True
```

## 1. Data (once)

The original study used the Kaggle mirrors of the PhysioNet/CinC Challenge 2021
databases. Either download them automatically:

```bat
pip install kaggle
:: put your API token in %USERPROFILE%\.kaggle\kaggle.json (kaggle.com -> Settings -> Create New Token)
python prepare_data.py --raw_root raw --out data --download
```

or unpack them yourself anywhere under `raw\` and run
`python prepare_data.py --raw_root raw --out data`. The script finds the folders
by name (Kaggle names such as `WFDB_Ningbo`, `Training_2`, or the PhysioNet
challenge names such as `WFDB_Ga`, `WFDB_CPSC2018_2`). It needs
`dx_mapping_scored.csv` (in the SNOMED mappings dataset) somewhere under `raw\`,
or pass `--mapping`.

Expect ~15 GB for the raw data and ~4.5 GB for `data\`. Check the printed counts:
the paper reports PTB-XL = 21,837 recordings and 30 classes.

## 2. Smoke test (≈ 15–30 min)

```bat
python run_all.py --smoke
```

300 recordings per centre, 2 epochs, 2 cycles/rounds, 1 seed, 50 bootstrap
resamples. It checks that every strategy and the analysis run end to end. The
numbers are meaningless; delete `results_smoke\` afterwards.

Then estimate the full run time on this machine (on mains power, Windows power
mode "Best performance"):

```bat
python benchmark_speed.py
```

On mains power (10 Nov 2026) it measured 84 ms per step: ~17 h per seed, ~3.5 days
for 5 seeds (on battery: 200 ms per step, ~9 days). Most of the time goes to CIIL, repeated FedAvg, CIIL 1 epoch and FedSGD.
The step time is dominated by Keras/PyTorch overhead, not GPU compute.

## 3. Full run

```bat
python run_all.py
```

7 strategies × 5 seeds, then the analysis.

**Stopping and starting.** Stop at any time with Ctrl+C (or shut the laptop
down) and run the same command again later:
- finished runs (`run_info.json` present) are skipped;
- an interrupted run resumes from its last checkpoint
  (`results_revision\<strategy>_seed<k>\checkpoint.pkl`), written after each
  centre (IIL, FedAvg), each cycle (CIIL, CIIL 1 epoch) and each FedSGD
  evaluation (every 200 rounds). The checkpoint holds the model weights, the Adam
  optimizer state, the best model so far, the history and the random-number
  states. At most one centre's training is repeated. CDS (~25 min) restarts.
- the log of a resumed run is appended to, and `run_info.json` records
  `resumed_from_checkpoint` and the total training time over all sessions.

A resumed run is a valid run of the same procedure, but not bit-identical to an
uninterrupted one (shuffling order and GPU non-determinism differ). Progress is printed per run; each run's
full log is in `results_revision\logs\`. Runs go seed by seed, so after the first
seed you can look at `python run_all.py --eval_only --seeds 1` while the rest trains.

Useful options: `--seeds 1 2 3` (fewer seeds), `--strategies cds fedavg`,
`--eval_only`, `--n_jobs 4` (bootstrap processes; reduce if RAM is short).

Keep the laptop on mains power with sleep disabled. With 16 GB RAM, close
other heavy programs: CDS holds all training data (~3.5 GB) twice in memory.

## What changed relative to the original code (commit 4e4be36)

Bugs that affected the submitted results:

1. **FedAvg / repeated FedAvg**: every centre got the *same* Keras model object,
   so "local" training was sequential and the "average" combined identical
   weights (the results were effectively IIL / CIIL without INCART). Now each
   centre trains its own copy from the global weights; aggregation is weighted
   by training set size; INCART is included.
2. **FedSGD**: the loss treated sigmoid outputs as logits (`from_logits=True`);
   gradients were summed; only 100–200 steps were taken. Now `from_logits=False`,
   weighted average, early stopping on validation AUROC (evaluated every 200
   rounds, patience 5, max 20,000 rounds).
3. **CIIL**: model selection now uses the mean validation AUROC of the
   end-of-cycle model, and the selected (not the last) model is evaluated.
4. **Keras 3 epoch bug**: with Keras 3, `fit(..., steps_per_epoch=n // batch_size)`
   on arrays makes every second epoch train on a single leftover batch. The
   `steps_per_epoch`/`validation_steps` arguments were removed, so each epoch uses
   all training data (including the last partial batch) and validation uses all
   validation data. (Under the original TF 2 / Keras 2 the last partial batch was
   dropped instead; the effect of this difference is negligible.)

Other changes: PTB-XL is used only once, after training; predictions, weights,
histories and timings are saved; data are stored as float32 (identical model
input); preprocessing in `prepare_data.py` reproduces the notebook exactly
(verified bit-for-bit against the notebook's `pad_and_resample`).

The original training scripts (`FedAVG.py`, `cyclic_*.py`, ...) are kept for
reference but are superseded by `run_strategy.py`; `cv_local_hospital.py` no
longer works with the new `centre.py`.

## Files

| File | Purpose |
|---|---|
| `prepare_data.py` | raw WFDB → `data/X_data_*.npy`, `y_data_*.npy`, `meta_*.csv`, `classes.csv` |
| `run_strategy.py` | trains one strategy with one seed, saves predictions |
| `run_all.py` | runs everything (resumable), then the analysis |
| `evaluate_with_ci.py` | metrics, bootstrap CIs, paired bootstrap and DeLong tests, LaTeX rows |
| `dataset_stats.py` | Supplementary Tables S1–S2, `ptbxl_meta.csv` |
| `plot_training_curves.py` | Supplementary Figure S1 |
| `src/centres/centre.py`, `src/models/model.py` | corrected, backend-agnostic versions |
