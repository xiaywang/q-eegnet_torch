import unittest
from random import shuffle


class KFoldCV:
    """
    Class for generating KFold CV
    """
    def __init__(self, n_splits):
        """
        n_splits must be larger or equal to 2.
        """
        assert n_splits >= 2
        self.n_splits = n_splits

    def split(self, x, y):
        """
        Generates the cross validation splits based on x and y
        Parameters:
         - x: t.tensor, size = [n, C, T]
         - y: t.tensor, size = [n]

        Returns:
         - Iterator, where each element is: (train_idx, validation_idx)
           - train_idx:       list of indices
           - validation_idx:  list of indices
        """

        assert x.shape[0] == y.shape[0]
        n = y.shape[0]

        # generate a list of all indices
        indices = list(range(n))

        # shuffle this list
        shuffle(indices)

        # split this into n_splits equally sized sets
        splits = []
        current_pos = 0
        for i in range(self.n_splits):
            split_size = n // self.n_splits
            if i + 1 == self.n_splits:
                split_size = n - ((n // self.n_splits) * (self.n_splits - 1))
            splits.append(indices[current_pos:current_pos + split_size])
            current_pos += split_size
        assert current_pos == n

        for i in range(self.n_splits):
            # generate training splits
            train_splits = set(range(self.n_splits))
            train_splits.remove(i)
            train_idx = []
            for split in train_splits:
                train_idx += splits[split]

            # generate validation splits
            validation_idx = splits[i]

            yield train_idx, validation_idx


class TestKFoldCV(unittest.TestCase):
    def test_kfold_cv(self):
        import torch as t
        from random import randint

        # repeat the test for K in [2, 10]
        for i in range(2, 11):
            x = t.zeros((randint(200, 400), 22, 1750))
            y = t.zeros(x.shape[0])
            cv = KFoldCV(n_splits=i)
            for test_idx, validation_idx in cv.split(x, y):
                # No duplicates are allowed in test set and validation set
                assert len(test_idx) == len(set(test_idx))
                assert len(validation_idx) == len(set(validation_idx))
                # No training sample is allowed in the validation set
                assert not set(test_idx).intersection(set(validation_idx))
                # number of samples must be equal to the length of x and y
                assert len(test_idx) + len(validation_idx) == y.shape[0]
                # only values between 0 and y.shape[0] are allowed
                assert 0 <= max(test_idx) < y.shape[0]
                assert 0 <= max(validation_idx) < y.shape[0]
                # all those assertions suffice to show that the split is a valid cross validation
