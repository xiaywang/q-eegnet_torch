"""
Miscellanious Functions for the EEGNet
"""

import unittest
import torch as t


def one_hot(x, n_classes=None):
    """
    Returns one_hot encoding of the input vector

    Parameters
     - x:         t.tensor, size=[n], dtype=t.long, integer values between 0 and n_classes
     - n_classes: number or None. if None, this value is determined automatically

    Returns: t.tensor, size_[n, n_classes], one_hot encoded
    """

    n = x.shape[0]

    if n_classes is None:
        n_classes = 1 + round(x.max().item())

    y = t.zeros(n, n_classes)
    # move to gpu if necessary
    if x.is_cuda:
        y = y.cuda()

    y.scatter_(1, x.reshape(n, 1), 1)
    return y


class TestOneHot(unittest.TestCase):
    def test_one_hot(self):
        from random import randint

        # size of testvector
        n = 256
        n_classes = 4

        # prepare testvector
        x = t.zeros(n).to(dtype=t.long)
        for i in range(n):
            x[i] = randint(0, n_classes-1)

        # generate one_hot encoding
        y = one_hot(x, n_classes=n_classes)

        # check output
        for i in range(n):
            for j in range(n_classes):
                if j == x[i]:
                    assert y[i, j] == 1
                else:
                    assert y[i, j] == 0

        # do the same thing without specifying n_classes
        y2 = one_hot(x)
        assert t.all(y == y2).item()

        # do the same for cuda vector
        yc = one_hot(x.cuda())
        yc = yc.cpu()
        assert t.all(y == y2).item()
