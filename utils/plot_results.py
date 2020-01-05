"""
Utility Functions to plot the results and store on disk
"""

import os
import itertools

import matplotlib.pyplot as plt
import torch as t
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix

from .misc import one_hot, class_decision


def generate_plots(subject, model, test_loader, loss, accuracy, lr=None, target_dir=None):
    """
    Generates all plots and stores them on the disk

    Parameters:
     - subject:     number between 1 and 9
     - model:       t.Module, trained model
     - test_loader: t.utils.data.DataLoader
     - loss:        t.tensor, size = [2, epochs], 0: training, 1: testing
     - accuracy:    t.tensor, size = [2, epochs], 0: training, 1: testing
     - target_dir:  string or os.path, if None, <current_file>/../results is used.
    """

    # make sure that the environment variables are set (to hide the unnecessary output)
    if "XDG_RUNTIME_DIR" not in os.environ:
        tmp_dir = "/tmp/runtime-eegnet"
        os.environ["XDG_RUNTIME_DIR"] = tmp_dir
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
            os.chmod(tmp_dir, 700)

    # necessary to compute the data first
    model.train(False)

    y_hat = None
    y = None
    for x_batch, y_batch in test_loader:
        output = model(x_batch)
        if y_hat is None and y is None:
            y_hat = output
            y = y_batch
        else:
            y_hat = t.cat((y_hat, output), axis=0)
            y = t.cat((y, y_batch), axis=0)

    y = y.cpu().detach()
    y_hat = y_hat.cpu().detach()

    # generate loss_accuracy plot
    plot_loss_accuracy(subject, loss, accuracy, lr, target_dir)

    # generate precision recall plot
    plot_precision_recall_curve(subject, y, y_hat, target_dir=target_dir)

    # generate confusion matrix
    plot_confusion_matrix(subject, y, y_hat, target_dir=target_dir)

    # It is probably not necessary to move y back to cuda, because the data is no longer used.
    # But I will do it anyways.
    y = y.cuda()


def plot_loss_accuracy(subject, loss, accuracy, lr=None, target_dir=None):
    """
    Generates a plot showing the evolution of the loss and the accuracy over all epochs.

    Parameters:
     - subject:    number, 1 <= subject <= 9
     - loss:       t.tensor, size = [2, epochs]
     - accuracy:   t.tensor, size = [2, epochs]
     - target_dir: string or os.path, if None, <current_file>/../results is used.
    """

    assert loss.shape == accuracy.shape

    # prepare filename
    filename = _get_filename(subject, "loss_acc", target_dir)

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
    plt.grid()

    if lr is not None:
        lr = lr.detach().numpy()
        lr_axis = loss_subfig.twinx()
        lr_axis.set_ylabel("Learning Rate")
        lr_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][2]
        lr_axis.plot(x, lr, label="Learning Rate", color=lr_color)

    loss_subfig.set_title("Loss")
    loss_subfig.set_xlabel("Epoch")
    loss_subfig.legend(loc="upper left")

    # do accuracy figure
    accuracy_subfig = fig.add_subplot(122)
    accuracy_subfig.plot(x, accuracy[0, :], label="training")
    accuracy_subfig.plot(x, accuracy[1, :], label="testing")
    accuracy_subfig.set_title("Accuracy")
    accuracy_subfig.set_xlabel("Epoch")
    accuracy_subfig.legend(loc="upper left")
    plt.grid()

    # save the image
    fig.savefig(filename, bbox_inches='tight')

    # close
    plt.close('all')


def plot_precision_recall_curve(subject, y, y_pred, n_classes=4, target_dir=None):
    """
    Generates a Precision-Recall curve and stores it.

    Parameters:
     - subject:    number of the subject, between 1 and 9
     - y:          t.tensor, size=[n_samples], the correct output
     - y_pred:     t.tensor, size=[n_samples, n_classes], prediction output
     - n_classes:  number of classes
     - target_dir: string or os.path, if None, <current_file>/../results is used.

    Returns: float, Average precision score, micro averaged over all classes
    """

    # prepare filename
    filename = _get_filename(subject, "loss_acc", target_dir)

    # prepare the data
    precision = {}
    recall = {}
    average_precision = {}

    y_one_hot = one_hot(y, n_classes=n_classes).detach().numpy()
    y_pred = y_pred.numpy()

    # compute precision and recall for each class
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_one_hot[:, i], y_pred[:, i])
        average_precision[i] = average_precision_score(y_one_hot[:, i], y_pred[:, i])

    # compute the averaged precision and recall for all classes
    precision['micro'], recall['micro'], _ = precision_recall_curve(y_one_hot.ravel(),
                                                                    y_pred.ravel())
    average_precision['micro'] = average_precision_score(y_one_hot, y_pred, average='micro')

    # plot the data
    plt.figure()
    plt.step(recall['micro'], precision['micro'], color='b', alpha=.2, where='post')
    plt.fill_between(recall['micro'], precision['micro'], step='post', alpha=.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([.0, 1.05])
    plt.xlim([.0, 1.0])
    plt.title('Average precision score, micro averaged over all classes: AP={0:0.2f}'
              .format(average_precision['micro']))
    plt.savefig('{}_precision_recall.png'.format(filename), bbox_incches='tight')
    plt.close('all')
    return average_precision['micro']


def plot_confusion_matrix(subject, y, y_pred, class_names=None, normalize=False, cmap=plt.cm.Blues,
                          target_dir=None):
    """
    Generates a Confusion Matrix plot and stores it on disk
    """
    # prepare filename
    filename = _get_filename(subject, "confusion", target_dir)

    # prepare class names
    if class_names is None:
        class_names = ["Left hand", "Right hand", "Both feet", 'Tongue']

    # generate confusion matrix
    y_decision = class_decision(y_pred)
    cm = confusion_matrix(y, y_decision)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Generate Plot
    plt.figure()
    fig, ax = plt.subplots()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    ax.set_ylim(len(cm) - 0.5, -0.5)
    plt.savefig(filename, bbox_inches='tight')
    plt.close('all')


def _get_filename(subject, name, target_dir=None):
    """
    Returns the requested filename including the path

    Parameters:
     - subject:    number of the subject, between 1 and 9
     - name:       name of the file
     - target_dir: path to the folder to store. If None, use the results folder in the project root.

    Return: String of the format: /path/to/target/folder/s{subject}_{name}.png
    """

    if target_dir is None:
        target_dir = os.path.dirname(os.path.realpath(__file__))
        target_dir = os.path.join(target_dir, '../results')
        target_dir = os.path.realpath(target_dir)
    return os.path.join(target_dir, f"s{subject}_{name}.png")
