"""
Implementation for early stopping for PyTorch models
"""

import torch as t
import numpy as np


class EarlyStopping():
    """
    Implementation for Early Stopping

    Always stores the best model in a file, which can be accessed at the end of all epochs.
    Invoke self.checkpoint() after every training step
    """

    def __init__(self, filename=".checkpoint.params", delta=0, criterion='loss'):
        """
        Constructor

        Parameters:
         - filename: string for the filename (can also be an absolute path)
         - delta: minimum change needed for the model to update on the disk
         - criterion: 'loss' or 'accuracy', which should be the indicator for the best model
        """
        assert criterion in ['loss', 'accuracy']
        self.best_accuracy = 0
        self.best_loss = np.inf
        self.best_epoch = 0
        self.delta = delta
        self.filename = filename
        self.criterion = criterion

    def checkpoint(self, model, loss, accuracy, epoch):
        """
        Stores the model if the accuracy is better.
        This function must be called every time an epoch is computed

        Parameters:
         - model:    t.nn.Module, current model
         - loss:     float, validation loss of the model
         - accuracy: float, validation accuracy of the model
         - epoch:    integer, current epoch of the model
        """
        is_better = False
        if self.criterion == 'loss' and loss < self.best_loss - self.delta:
            is_better = True
        if self.criterion == 'accuracy' and accuracy > self.best_accuracy + self.delta:
            is_better = True

        if is_better:
            # update the metadata
            self.best_loss = loss
            self.best_accuracy = accuracy
            self.best_epoch = epoch
            # store to disk
            t.save(model.state_dict(), self.filename)

    def use_best_model(self, model):
        """
        Overwrites the model with the parameters and returns it

        Parameters:
         - model: t.nn.Module, all weights will probably be overwritten

        Returns: (model, loss, accuracy, epoch)
        """
        model.load_state_dict(t.load(self.filename))
        model.eval()
        return model, self.best_loss, self.best_accuracy, self.best_epoch
