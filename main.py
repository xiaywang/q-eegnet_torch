"""
Main function for EEGnet, work in progress...
"""

import torch as t

from eegnet_controller import train_subject_specific, train_subject_specific_cv
from utils.metrics import metrics_to_csv


DO_CV = True
N_EPOCHS = 500


sum_accuracy = 0.0

metrics = t.zeros((9, 4))

for subject in range(1, 10):

    epochs = N_EPOCHS

    if DO_CV:
        # First, do cross validation to determine optimal number of epochs
        # Afterwards, do normal training
        _, _, best_epoch = train_subject_specific_cv(subject, epochs=epochs, plot=False)
        epochs = best_epoch

    _model, subject_metrics = train_subject_specific(subject, epochs=epochs)
    metrics[subject-1, :] = subject_metrics[0, :]

metrics_to_csv(metrics)

print(f"\nAverage Accuracy: {metrics[:,0].mean()}")
