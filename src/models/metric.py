import keras.backend as K
import tensorflow as tf
from sklearn.metrics import roc_auc_score

class ROCAUCMetricMultiLabel(tf.keras.metrics.Metric):
    def __init__(self, name='AUROC', average='macro', **kwargs):
        super(ROCAUCMetricMultiLabel, self).__init__(name=name, **kwargs)
        self.average = average
        # Initialize placeholders for true labels and predicted values
        self.true_labels = self.add_weight(name='true_labels', shape=(0, 0), dtype=tf.float32, initializer='zeros', aggregation=None)
        self.predictions = self.add_weight(name='predictions', shape=(0, 0), dtype=tf.float32, initializer='zeros', aggregation=None)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Reshape y_true and y_pred to ensure they are 2D arrays (for multi-label classification)
        y_true = tf.reshape(y_true, (-1, y_true.shape[-1]))
        y_pred = tf.reshape(y_pred, (-1, y_pred.shape[-1]))

        # Concatenate current batch values to the stored values
        self.true_labels.assign(tf.concat([self.true_labels, y_true], axis=0))
        self.predictions.assign(tf.concat([self.predictions, y_pred], axis=0))

    def result(self):
        # Compute ROC AUC score using Scikit-learn for multi-label classification
        true_labels = self.true_labels.numpy()
        predictions = self.predictions.numpy()

        # Handle cases where some labels may not have positive examples
        try:
            roc_auc = roc_auc_score(true_labels, predictions, average=self.average)
        except ValueError:
            # This happens if only one class is present in y_true for some labels, return 0 in that case.
            roc_auc = 0.0
        
        return tf.constant(roc_auc, dtype=tf.float32)

    def reset_states(self):
        # Reset the state of the metric between epochs
        self.true_labels.assign(tf.zeros((0, 0), dtype=tf.float32))
        self.predictions.assign(tf.zeros((0, 0), dtype=tf.float32))
