# REVISION version of centre.py. Changes from the original (commit 4e4be36) are
# marked "REVISION". In short:
#   * Keras 3 (`import keras`), so it runs on the PyTorch backend (GPU on Windows).
#   * fit() no longer gets steps_per_epoch / validation_steps. In Keras 3 an array
#     input with steps_per_epoch = n // batch_size leaves the last partial batch in
#     the iterator, so every second epoch trains on ONE batch only. Every epoch now
#     covers the whole training set, and validation uses the whole validation set.
#   * FedSGD loss uses from_logits=False (the model ends in a sigmoid); gradients
#     are averaged (weighted by training set size), not summed; the generator
#     reshuffles every pass and uses batch size 32.
#   * Unique temporary checkpoint folder per process; data stored as float32 and
#     the unsplit copy is freed after the split (to fit in 16 GB RAM).
#   * TRAIN_CONFIG lets run_strategy.py shorten training for a smoke test.
import os
import shutil
import tempfile

import keras
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from src.models.model import build_iception_model

# REVISION: max_epochs/patience as in the original (100/5); changed only for --smoke runs.
TRAIN_CONFIG = {"max_epochs": 100, "patience": 5, "verbose": 2}


def _fit_to_convergence(model, X_train, y_train, X_val, y_val, batch_size, log_filename):
    """Train with early stopping on validation AUROC and restore the best epoch
    (same callbacks as the original train_to_convergence)."""
    temp_dir = tempfile.mkdtemp(prefix="ecgfl_")  # REVISION: unique per process
    checkpoint = os.path.join(temp_dir, "model_weights_temp.weights.h5")
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_AUROC", mode="max", patience=TRAIN_CONFIG["patience"],
                                      verbose=1, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(checkpoint, monitor="val_AUROC", mode="max", save_best_only=True,
                                        save_weights_only=True, verbose=0),
        keras.callbacks.CSVLogger(log_filename),
    ]
    model.fit(X_train, y_train, epochs=TRAIN_CONFIG["max_epochs"], batch_size=batch_size,
              validation_data=(X_val, y_val), callbacks=callbacks, shuffle=True,
              verbose=TRAIN_CONFIG["verbose"])
    model.load_weights(checkpoint)
    shutil.rmtree(temp_dir, ignore_errors=True)
    print("Training complete")


def compute_gradients(model, X_batch, y_batch):
    """REVISION: backend-agnostic gradient of the focal loss for one batch
    (the model outputs probabilities, so from_logits=False)."""
    loss_fn = keras.losses.BinaryFocalCrossentropy(from_logits=False)
    x = keras.ops.convert_to_tensor(X_batch.astype(np.float32))
    y = keras.ops.convert_to_tensor(y_batch.astype(np.float32))
    backend = keras.backend.backend()
    if backend == "torch":
        for v in model.trainable_variables:
            v.value.grad = None
        loss = loss_fn(y, model(x, training=True))
        loss.backward()
        return [v.value.grad.detach().clone() for v in model.trainable_variables]
    if backend == "tensorflow":
        import tensorflow as tf
        with tf.GradientTape() as tape:
            loss = loss_fn(y, model(x, training=True))
        return tape.gradient(loss, model.trainable_variables)
    raise NotImplementedError(f"FedSGD is not implemented for the {backend} backend")


class LocalHospital:
    def __init__(self, name, X_path, y_path, subsample=None):
        self.name = name
        # REVISION: float32 (the model computes in float32 anyway)
        X_data = np.moveaxis(np.load(X_path), 1, -1).astype(np.float32)
        y_data = np.load(y_path).astype(np.float32)
        if subsample and subsample < len(y_data):  # REVISION: smoke tests only (random, fixed subset)
            idx = np.sort(np.random.default_rng(0).choice(len(y_data), subsample, replace=False))
            X_data, y_data = X_data[idx], y_data[idx]
        self.val_auroc = []
        print(name, X_data.shape, y_data.shape)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_data, y_data, test_size=0.20, random_state=42)  # 20% val, same split as the original
        del X_data, y_data  # REVISION: keep only the split copies

    def load_model(self, model):
        self.model = model

    def batch_size(self):
        return 10 if self.X_train.shape[0] < 100 else 32  # an exception for very small datasets

    def train_one_epoch(self):
        # REVISION: no steps_per_epoch (see top of file)
        self.model.fit(self.X_train, self.y_train, epochs=1, batch_size=self.batch_size(),
                       shuffle=True, verbose=TRAIN_CONFIG["verbose"])

    def predict_val(self):
        val_hat = self.model.predict(self.X_val, verbose=0)
        if 0 < self.y_val.sum() < self.y_val.size:
            val_auroc = roc_auc_score(self.y_val.ravel(), val_hat.ravel())
        else:  # REVISION: AUROC undefined (no positive labels in this validation set)
            val_auroc = float("nan")
        self.val_auroc.append(val_auroc)
        if np.isnan(val_auroc):
            return None, None, val_auroc
        fpr, tpr, _ = roc_curve(self.y_val.ravel(), val_hat.ravel())
        return fpr, tpr, val_auroc

    def predict_val_raw(self):
        """Return (y_val, y_hat) for the current weights, for threshold selection."""
        return self.y_val, self.model.predict(self.X_val, verbose=0)

    def n_train(self):
        return self.X_train.shape[0]

    def get_weights(self):
        return self.model.get_weights()

    def get_data(self, split="train"):
        if split == "train":
            return self.X_train, self.y_train
        if split == "val":
            return self.X_val, self.y_val

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def train_to_convergence(self, log_filename):
        _fit_to_convergence(self.model, self.X_train, self.y_train, self.X_val, self.y_val,
                            self.batch_size(), log_filename)

    @staticmethod
    def batch_generator(batch_size, gen_data):
        while True:
            batch = [next(gen_data) for _ in range(batch_size)]
            yield np.stack([b[0] for b in batch]), np.stack([b[1] for b in batch])

    @staticmethod
    def generate_data(X, y):
        while True:
            # REVISION: reshuffle on every pass (the original iterated in file order)
            for i in np.random.permutation(len(y)):
                yield X[i], y[i]

    def set_generator(self):
        self.dat_gen = self.generate_data(self.X_train, self.y_train)
        # REVISION: batch size 32, as in all other strategies (was 30)
        self.bgen = self.batch_generator(32, self.dat_gen)

    def train_one_batch(self):
        X_batch, y_batch = next(self.bgen)
        return compute_gradients(self.model, X_batch, y_batch)


class CentralModelDistributor:
    def __init__(self, input_shape, output_shape):
        self.model = build_iception_model(input_shape, output_shape)
        self.weight_list = []
        self.grads_list = []
        self.optimizer = keras.optimizers.Adam()

    def get_model(self):
        return self.model

    def update_weight_list(self, new_weights):
        self.weight_list.append(new_weights)

    def return_avg_weights(self):
        summed_weights = [sum(w) for w in zip(*self.weight_list)]
        avg_weight = [w / len(self.weight_list) for w in summed_weights]
        self.weight_list = []
        return avg_weight

    def load_weights_to_model(self, weights):
        self.model.set_weights(weights)

    def get_weights_from_model(self):
        return self.model.get_weights()

    def update_grads_list(self, new_grads):
        self.grads_list.append(new_grads)

    def apply_weighted_avg_grads(self, sizes):
        """REVISION: average the gradients, weighted by local training set size
        (the original summed them), and apply with the server-side Adam optimizer."""
        w = np.asarray(sizes, dtype=np.float64) / np.sum(sizes)
        avg_grads = [sum(float(wk) * g for wk, g in zip(w, grads)) for grads in zip(*self.grads_list)]
        self.optimizer.apply_gradients(zip(avg_grads, self.model.trainable_variables))
        self.grads_list = []

    @staticmethod
    def weighted_avg_weights(weight_lists, sizes):
        """REVISION: FedAvg aggregation weighted by local training set size
        (McMahan et al. 2017). Includes non-trainable batch-norm statistics."""
        w = np.asarray(sizes, dtype=np.float64) / np.sum(sizes)
        return [sum(wk * layer for wk, layer in zip(w, layers)).astype(layers[0].dtype)
                for layers in zip(*weight_lists)]


class ExternalValidationHospital:
    """External test centre (PTB-XL). Only used for prediction after training."""

    def __init__(self, name, X_path, y_path, subsample=None):
        self.name = name
        self.X_data = np.moveaxis(np.load(X_path), 1, -1).astype(np.float32)
        self.y_data = np.load(y_path).astype(np.float32)
        if subsample and subsample < len(self.y_data):
            idx = np.sort(np.random.default_rng(0).choice(len(self.y_data), subsample, replace=False))
            self.X_data, self.y_data = self.X_data[idx], self.y_data[idx]
        print(name, self.X_data.shape, self.y_data.shape)

    def load_model(self, model):
        self.model = model

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    def predict(self):
        y_hat = self.model.predict(self.X_data, verbose=0)
        fpr, tpr, _ = roc_curve(self.y_data.ravel(), y_hat.ravel())
        return fpr, tpr, roc_auc_score(self.y_data.ravel(), y_hat.ravel())


class CentralTrainer:
    """Collaborative data sharing: pools the centres' training and validation data."""

    def __init__(self, input_shape, output_shape):
        self.model = build_iception_model(input_shape, output_shape)
        self.X_train_parts, self.y_train_parts, self.X_val_parts, self.y_val_parts = [], [], [], []

    def load_train_data_from_centre(self, X_data, y_data):
        self.X_train_parts.append(X_data)
        self.y_train_parts.append(y_data)

    def load_val_data_from_centre(self, X_data, y_data):
        self.X_val_parts.append(X_data)
        self.y_val_parts.append(y_data)

    def train_to_convergence(self, log_filename):
        X_train, y_train = np.concatenate(self.X_train_parts), np.concatenate(self.y_train_parts)
        X_val, y_val = np.concatenate(self.X_val_parts), np.concatenate(self.y_val_parts)
        self.X_train_parts = self.y_train_parts = self.X_val_parts = self.y_val_parts = None
        _fit_to_convergence(self.model, X_train, y_train, X_val, y_val, 32, log_filename)

    def get_model(self):
        return self.model
