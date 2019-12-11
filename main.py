"""
Main function for EEGnet, work in progress...
"""

import torch as t
from tqdm import tqdm

from eegnet_controller import train_subject_specific, train_subject_specific_cv
from utils.metrics import metrics_to_csv


DO_CV = False
N_EPOCHS = 500

BENCHMARK = False
N_TRIALS = 20


def run(do_cv=False, epochs=500, export=True, silent=False):
    metrics = t.zeros((9, 4))
    for subject in range(1, 10):
        if do_cv:
            # First, do cross validation to determine optimal number of epochs
            # Afterwards, do normal training
            _, _, best_epoch = train_subject_specific_cv(subject, epochs=epochs, silent=silent,
                                                         plot=False)
            epochs = best_epoch

        _model, subject_metrics = train_subject_specific(subject, epochs=epochs, silent=silent,
                                                         plot=export)
        metrics[subject-1, :] = subject_metrics[0, :]

    if export:
        metrics_to_csv(metrics)

    return metrics


def main():
    if BENCHMARK:
        print("Benchmarking...")

        metrics = t.zeros((N_TRIALS, 9, 4))
        for i in tqdm(range(N_TRIALS), ascii=True):
            metrics[i, :, :] = run(do_cv=DO_CV, epochs=N_EPOCHS, export=False, silent=True)

        # generate the average over the first dimension
        # avg_metrics is of size 9, 4, with all 4 scores for all 9 subjects averaged over all trials
        avg_metrics = metrics.mean(axis=0)
        var_metrics = metrics.var(axis=0)

        # store the results
        metrics_to_csv(avg_metrics, "benchmark_mean_metrics.csv")
        metrics_to_csv(var_metrics, "benchmark_var_metrics.csv")

        print(f"Total Average Accuracy: {avg_metrics[:, 0].mean()}\n")
        for i in range(0, 9):
            print(f"subject {i+1}: accuracy = {avg_metrics[i, 0]} +- {var_metrics[i, 0]}")

    else:
        # normal procedure
        metrics = run(do_cv=DO_CV, epochs=N_EPOCHS, export=True, silent=False)
        print(f"\nAverage Accuracy: {metrics[:,0].mean()}")


if __name__ == "__main__":
    main()
