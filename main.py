"""
Main function for EEGnet, work in progress...
"""

import torch as t

from eegnet_controller import train_subject_specific
from utils.metrics import metrics_to_csv

sum_accuracy = 0.0

metrics = t.zeros((9, 4))

for subject in range(1, 10):
    _model, subject_metrics = train_subject_specific(subject, epochs=500)
    metrics[subject-1, :] = subject_metrics[0, :]

metrics_to_csv(metrics)

print(f"\nAverage Accuracy: {metrics[:,0].mean()}")
