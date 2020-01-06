"""
Main function for EEGnet, work in progress...
"""

from functools import reduce

import torch as t
import numpy as np
from tqdm import tqdm

from eegnet_controller import train_subject_specific, train_subject_specific_cv, \
    train_subject_specific_quant
from eegnet_controller import test_model_from_keras
from utils.metrics import metrics_to_csv
from utils.misc import product_dict
from utils.plot_results import plot_loss_accuracy


QUANTIZED = True
DO_CV = False
N_EPOCHS = 500

BENCHMARK = False
N_TRIALS = 20

GRID_SEARCH = False

TEST_KERAS_MODEL = False


def run(do_cv=False, epochs=N_EPOCHS, quantized=QUANTIZED, export=True, silent=False):
    """
    Does one complete run over all data
    """
    metrics = t.zeros((9, 4))
    loss_history = t.zeros((9, 2, epochs))
    acc_history = t.zeros((9, 2, epochs))
    for subject in range(1, 10):
        if do_cv:
            # First, do cross validation to determine optimal number of epochs
            # Afterwards, do normal training
            _, _, best_epoch = train_subject_specific_cv(subject, epochs=epochs, silent=silent,
                                                         plot=False)
            epochs = best_epoch

        if quantized:
            _model, subject_metrics, history = \
                train_subject_specific_quant(subject, epochs=epochs, silent=silent, plot=export)
        else:
            _model, subject_metrics, history = \
                train_subject_specific(subject, epochs=epochs, silent=silent, plot=export)
        loss, acc = history
        loss_history[subject-1, :, :] = loss
        acc_history[subject-1, :, :] = acc
        metrics[subject-1, :] = subject_metrics[0, :]

    if export:
        metrics_to_csv(metrics)

    return metrics, (loss_history, acc_history)


def grid_search(grid_params, epochs=500, silent=False):
    """
    Applies GridSearch to determine optimal hyperparameter using Cross Validation

    Parameters:
     - grid_params: Dictionary, with keys as parameter names, and values as a list
     - epochs:      Number of epochs to train
     - silent:      Bool, if True, hide all output (progress bar)

    Returns: List of tuples: (parameters as dict, validation accuracy)
    """
    # compute total number of iterations
    n_iter = reduce(lambda x, y: x * y, [len(param_list) for param_list in grid_params.values()])

    # prepare result
    # Type: list of tuple: (parameters as dict, accuracy)
    result = []

    # iterate over all possible combinations
    with tqdm(total=n_iter * 9, ascii=True, desc='Grid Search') as pbar:
        for params in product_dict(**grid_params):
            accuracy = t.zeros((9,))
            for subject in range(1, 9):
                _, metrics, _ = train_subject_specific_cv(subject, epochs=epochs, silent=True,
                                                          plot=False, **params)
                accuracy[subject-1] = metrics[0, 0]

                # update the progress bar
                pbar.update()

            # store the average accuracy
            result.append((params, accuracy.mean()))

    return result


def main():
    """
    Main function used for testing
    """
    if TEST_KERAS_MODEL:
        acc = np.zeros((9,))
        for subject in range(1, 10):
            _, accuracy = test_model_from_keras(subject)
            acc[subject-1] = accuracy
            print(f"Subject {subject}: accuracy = {accuracy}")

        print(f"\nAverage Accuracy: {acc.mean()}")

    elif BENCHMARK:
        metrics = t.zeros((N_TRIALS, 9, 4))
        average_loss = t.zeros((9, 2, N_EPOCHS))
        average_acc = t.zeros((9, 2, N_EPOCHS))
        for i in tqdm(range(N_TRIALS), ascii=True, desc='Benchmark'):
            metrics[i, :, :], history = run(do_cv=DO_CV, epochs=N_EPOCHS, export=False, silent=True)
            loss, acc = history
            average_loss += loss
            average_acc += acc

        # generate the average over the first dimension
        # avg_metrics is of size 9, 4, with all 4 scores for all 9 subjects averaged over all trials
        avg_metrics = metrics.mean(axis=0)
        std_metrics = metrics.std(axis=0)

        # For the overall score, first average along all subjects.
        # For standard deviation, average all standard deviations of all subjects
        overall_avg_acc = avg_metrics[:, 0].mean()
        overall_std_acc = std_metrics[:, 0].mean()

        # store the results
        metrics_to_csv(avg_metrics, filename="benchmark_mean_metrics.csv")
        metrics_to_csv(std_metrics, filename="benchmark_std_metrics.csv")

        print(f"Total Average Accuracy: {overall_avg_acc:.4f} +- {overall_std_acc:.4f}\n")
        for i in range(0, 9):
            print(f"subject {i+1}: accuracy = {avg_metrics[i, 0]:.4f} +- {std_metrics[i, 0]:.4f}")

        # average out the history
        average_loss /= N_TRIALS
        average_acc /= N_TRIALS
        for i in range(9):
            plot_loss_accuracy(i+1, average_loss[i, :, :], average_acc[i, :, :])

    elif GRID_SEARCH:
        # parameters to search
        grid_params = {
            'lr': [0.02, 0.01, 0.005],
            'lr_decay': [0.5, 0.2, 0.1]
        }

        # do the grid search
        results = grid_search(grid_params, epochs=N_EPOCHS, silent=False)

        # print the results and get the best accuracy
        best_accuracy = -1
        best_params = {}
        for params, accuracy in results:
            print(f"{params}:\taccuracy={accuracy}")
            if best_accuracy < accuracy:
                best_accuracy = accuracy
                best_params = params

        # Print the best parameters
        print(f"\nBest Parameters: {best_params}")

    else:
        # normal procedure
        metrics, _ = run(do_cv=DO_CV, epochs=N_EPOCHS, export=True, silent=False)
        print(f"\nAverage Accuracy: {metrics[:,0].mean()}")


if __name__ == "__main__":
    main()
