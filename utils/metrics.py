"""
Computes metrics and exports to csv
"""

import os

import numpy as np
import torch as t

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from .misc import class_decision


def get_metrics_from_model(model, test_loader):
    """
    computes metrics for result given a model and the training data

    Parameters:
     - model:       t.nn.Module
     - test_loader: t.utils.data.DataLoader

    Returns: t.tensor, size=[1, 4]: accuracy, precision, recall, f1
    """
    use_cuda = model.is_cuda()
    y_hat = None
    y = None
    for x_batch, y_batch in test_loader:
        if use_cuda:
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
        output = model(x_batch)
        if y_hat is None and y is None:
            y_hat = output
            y = y_batch
        else:
            y_hat = t.cat((y_hat, output), axis=0)
            y = t.cat((y, y_batch), axis=0)

    y = y.cpu().detach()
    y_hat = y_hat.cpu().detach()

    return get_metrics(y, y_hat)


def get_metrics(y, y_pred):
    """
    computes metrics for result for a single subject

    Parameters:
     - y:          t.tensor, size=[n_samples], the correct output
     - y_pred:     t.tensor, size=[n_samples, n_classes], prediction output

    Returns: t.tensor, size=[1, 4]: accuracy_score, precision_score, recall_score, f1_score
    """

    y_decision = class_decision(y_pred)

    metrics = t.zeros((1, 4))
    metrics[0, 0] = accuracy_score(y, y_decision)
    metrics[0, 1] = precision_score(y, y_decision, average='micro')
    metrics[0, 2] = recall_score(y, y_decision, average='micro')
    metrics[0, 3] = f1_score(y, y_decision, average='micro')

    return metrics


def metrics_to_csv(metrics, header=True, target_dir=None, filename='metrics.csv'):
    """
    Stores metrics as csv file

    Parameters:
     - metrics:    t.tensor, size=[n_subjects, 4]
     - header:     boolean, if True, add a heading row to the csv file
     - target_dir: path to the folder to store. If None, use the results folder in the project root.
     - filename:   name of the file to be stored
    """
    filename = _get_filename(filename, target_dir)
    header_row = ''
    if header:
        header_row = 'accuracy,precision,recall,f1'

    np.savetxt(filename, metrics, delimiter=',', header=header_row)


def _get_filename(name, target_dir=None):
    """
    Returns the filename for the metrics.csv file

    Parameters:
     - name:       name of the file (.csv is appended if not given)
     - target_dir: path to the folder to store. If None, use the results folder in the project root.

    Return: String of the format: /path/to/target/folder/{name}
    """

    if target_dir is None:
        target_dir = os.path.dirname(os.path.realpath(__file__))
        target_dir = os.path.join(target_dir, '../results')
        target_dir = os.path.realpath(target_dir)

    if not name.endswith("csv"):
        name += ".csv"

    return os.path.join(target_dir, name)
