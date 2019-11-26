'''
Loads the dataset 2a of the BCI Competition IV
available on http://bnci-horizon-2020.eu/database/data-sets
'''

from os import path
import numpy as np
import scipy.io as sio
import torch as t

__author__ = "Michael Hersche and Tino Rellstab, modified by Tibor Schneider"
__email__ = "herschmi@ethz.ch, tinor@ethz.ch, sctibor@ethz.ch"


DATA_PATH = "/scratch/sem19h24/BCI_IV_2a/mat"


def get_data(subject, training, data_path=None):
    """
    Loads the dataset 2a of the BCI Competition IV
    available on http://bnci-horizon-2020.eu/database/data-sets

    arguments:
     - subject:   number of subject in [1, .. ,9]
     - training:  if True, load training data
                  if False, load testing data
     - data_path: String, path to the BCI IV 2a dataset (.mat files)

    Returns: data_return,  numpy matrix, size = NO_valid_trial x 22 x 1750
             class_return, numpy matrix,	size = NO_valid_trial
    """

    if data_path is None:
        data_path = DATA_PATH

    NO_channels = 22
    NO_tests = 6 * 48
    Window_Length = 7 * 250

    class_return = np.zeros(NO_tests)
    data_return = np.zeros((NO_tests, NO_channels, Window_Length))

    NO_valid_trial = 0
    if training:
        a = sio.loadmat(path.join(data_path, 'A0' + str(subject) + 'T.mat'))
    else:
        a = sio.loadmat(path.join(data_path, 'A0' + str(subject) + 'E.mat'))
    a_data = a['data']
    for ii in range(0, a_data.size):
        a_data1 = a_data[0, ii]
        a_data2 = [a_data1[0, 0]]
        a_data3 = a_data2[0]
        a_X = a_data3[0]
        a_trial = a_data3[1]
        a_y = a_data3[2]
        # a_fs = a_data3[3]
        # a_classes = a_data3[4]
        a_artifacts = a_data3[5]
        # a_gender = a_data3[6]
        # a_age = a_data3[7]
        for trial in range(0, a_trial.size):
            if a_artifacts[trial] == 0:
                range_a = int(a_trial[trial])
                range_b = range_a + Window_Length
                data_return[NO_valid_trial, :, :] = np.transpose(a_X[range_a:range_b, :22])
                class_return[NO_valid_trial] = int(a_y[trial])
                NO_valid_trial += 1

    return data_return[0:NO_valid_trial, :, :], class_return[0:NO_valid_trial]


def as_tensor(samples, labels):
    """
    Returns the data as t.tensor, with the correct data type and on the GPU if available

    Parameters:
     - samples: np.ndarray, size = [s, C, T]
     - labels:  np.ndarray, size = [s]

    Returns:
     - samples: t.tensor, size = [s, C, T], dtype = float
     - labels:  t.tensor, size = [s],       dtype = long
    """

    x = t.tensor(samples).to(dtype=t.float)
    y = t.tensor(labels).to(dtype=t.long) - 1 # labels are from 1 to 4, but torch expects 0 to 3

    # move data to cuda if device is available
    if t.cuda.is_available():
        x = x.cuda()
        y = y.cuda()

    return x, y


def as_data_loader(samples, labels, batch_size=32, shuffle=True):
    """
    Returns the data as a t.utils.data.DataLoader.
    Moves data to device if available

    arguments:
     - samples: np.ndarray, size = [s, C, T]
     - labels:  np.ndarray, size = [s]

    Returns: t.utils.data.Dataloader
    """
    x, y = as_tensor(samples, labels)
    dataset = t.utils.data.TensorDataset(x, y)
    loader = t.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader
