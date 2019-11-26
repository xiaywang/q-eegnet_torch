from os import path

import matplotlib.pyplot as plt
import torch as t


TARGET_DIR = "/home/sem19h24/project/EEGnet_torch/results"


def plot_loss_accuracy(subject, loss, accuracy, target_dir=None):
    """
    Generates a plot showing the evolution of the loss and the accuracy over all epochs.

    Parameters:
     - subject:    number, 1 <= subject <= 9
     - loss:       t.tensor, size = [epochs]
     - accuracy:   t.tensor, size = [epochs]
     - target_dir: string, if None, <current_file>/../results is used.
    """

    assert loss.shape == accuracy.shape

    # prepare filename
    if target_dir is None:
        target_dir = path.dirname(path.realpath(__file__)).join("../results")
        target_dir = TARGET_DIR
    filename = path.join(target_dir, f"s{subject}_loss_acc.png")

    # prepare data
    x = t.tensor(range(loss.shape[1]))
    loss = loss.detach().numpy()
    accuracy = accuracy.detach().numpy()

    # prepare the plot
    fig = plt.figure(figsize=(20, 10))

    # do loss figure
    loss_subfig = fig.add_subplot(121)
    loss_subfig.plot(x, loss[0, :], label="training")
    loss_subfig.plot(x, loss[1, :], label="testing")
    loss_subfig.set_title("Loss")
    loss_subfig.set_xlabel("Epoch")
    loss_subfig.legend(loc="upper left")
    plt.grid()

    # do accuracy figure
    accuracy_subfig = fig.add_subplot(122)
    accuracy_subfig.plot(x, accuracy[0,:], label="training")
    accuracy_subfig.plot(x, accuracy[1,:], label="testing")
    accuracy_subfig.set_title("Accuracy")
    accuracy_subfig.set_xlabel("Epoch")
    accuracy_subfig.legend(loc="upper left")
    plt.grid()

    # save the image
    fig.savefig(filename, bbox_inches='tight')

    # close
    plt.close('all')
