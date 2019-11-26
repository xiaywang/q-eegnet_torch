"""
Miscellanious Functions for the EEGNet
"""

import torch as t


def one_hot(x, n_classes=None):
    """
    Returns one_hot encoding of the input vector

    Parameters
     - x:         torch.tensor, size=[n], dtype=t.long, integer values between 0 and n_classes
     - n_classes: number or None. if None, this value is determined automatically

    Returns: torch.tensor, size_[n, n_classes], one_hot encoded
    """

    n = x.shape[0]

    if n_classes is None:
        n_classes = 1 + round(x.max().item())

    y = t.zeros(n, n_classes)
    y.scatter_(1, x.reshape(n, 1), 1)
    return y
