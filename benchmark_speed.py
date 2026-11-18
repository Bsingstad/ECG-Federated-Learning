#!/usr/bin/env python

# Measures the training speed on this machine and estimates the duration of the
# full re-run. Run it on mains power with Windows power mode "Best performance":
#
#   python benchmark_speed.py
#
# Step counts per seed are estimated from the epoch counts in the original logs
# (reports/results): CDS 10 epochs; IIL/FedAvg 2-17 epochs per centre; CIIL and
# repeated FedAvg ~10 epochs per centre per cycle/round; CIIL 1 epoch 100 cycles;
# FedSGD at most 20,000 rounds x 6 centres. They are rough (+-50%).

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import time

import keras
import numpy as np

from src.models.model import build_iception_model

# training recordings (80%) per centre: INCART, PTB, Chapman, Ningbo, Georgia, CPSC
N_TRAIN = {"stpeter": 59, "ptbdiag": 413, "chapman": 8198, "ningbo": 27924, "georgia": 8275, "chinaphys": 8264}
TOTAL = sum(N_TRAIN.values())
IIL_EPOCHS = {"stpeter": 2, "ptbdiag": 7, "chapman": 17, "ningbo": 14, "georgia": 7, "chinaphys": 8}
CIIL_EPOCHS = {"stpeter": 100, "ptbdiag": 132, "chapman": 96, "ningbo": 94, "georgia": 82, "chinaphys": 114}
iil = sum(IIL_EPOCHS[k] * n for k, n in N_TRAIN.items())
ciil = sum(CIIL_EPOCHS[k] * n for k, n in N_TRAIN.items())
RECORDINGS = {"cds": 10 * TOTAL, "iil": iil, "fedavg": iil, "ciil": ciil, "fedavg_repeated": ciil,
              "ciil_1epoch": 100 * TOTAL}
FEDSGD_STEPS = 20000 * 6

model = build_iception_model((1000, 12), 30)
X = np.random.randn(3200, 1000, 12).astype("float32")
y = (np.random.rand(3200, 30) < 0.1).astype("float32")
model.fit(X[:320], y[:320], epochs=1, batch_size=32, verbose=0)  # warm-up
start = time.time()
model.fit(X, y, epochs=1, batch_size=32, verbose=0)
step = (time.time() - start) / 100
start = time.time()
model.predict(X, verbose=0)
pred_per_rec = (time.time() - start) / len(X)

print(f"Keras {keras.__version__}, backend {keras.backend.backend()}: {1000 * step:.0f} ms per training step (batch 32)")
hours = {}
for s, recs in RECORDINGS.items():
    epochs_equiv = recs / TOTAL
    hours[s] = (recs / 32 * step + epochs_equiv * 0.25 * TOTAL * pred_per_rec) / 3600  # + validation
hours["fedsgd"] = (FEDSGD_STEPS * step + 100 * 0.25 * TOTAL * pred_per_rec) / 3600
for s, h in hours.items():
    print(f"  {s:16s} ~{h:5.1f} h")
total = sum(hours.values())
print(f"One seed: ~{total:.0f} h. 3 seeds: ~{3 * total / 24:.1f} days. 5 seeds: ~{5 * total / 24:.1f} days.")
