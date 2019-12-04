import torch as t

from tqdm import tqdm

from eegnet import EEGNet
from utils.get_data import get_data, as_data_loader, as_tensor
from utils.kfold_cv import KFoldCV
from utils.plot_results import generate_plots
from utils.metrics import get_metrics_from_model
from utils.misc import class_decision
from utils.early_stopping import EarlyStopping


def train_subject_specific_cv(subject, n_splits=4, epochs=500, batch_size=32, lr=0.001,
                              progress=True, plot=True):
    """
    Trains a subject specific model for the given subject, using K-Fold Cross Validation

    Parameters:
     - subject: Integer in the Range 1 <= subject <= 9
     - n_splits: Number of splits for K-Fold CV
     - epochs: Number of epochs to train
     - batch_size: Batch Size
     - lr: Learning Rate
     - progress: bool, if True, displays a progress bar
     - plot: bool, if True, generates plots

    Returns: (models, metrics)
     - models:  List of t.nn.Module, size = [n_splits]
     - metrics: t.tensor, size=[1, 4], accuracy, precision, recall, f1
    """

    # load the raw data
    samples, labels = get_data(subject, training=True)
    samples, labels = as_tensor(samples, labels)

    # prepare the models
    models = [EEGNet(T=len(labels)) for _ in range(n_splits)]

    # prepare KFold
    kfcv = KFoldCV(n_splits)
    split = 0

    # prepare result
    loss = t.zeros((epochs, ))
    accuracy = t.zeros((epochs, ))

    # prepare progress bar
    if progress:
        pbar = tqdm(total=n_splits * epochs, desc=f"Subject {subject}", ascii=True)

    for split, indices in enumerate(kfcv.split(samples, labels)):
        # generate dataset
        train_idx, val_idx = indices
        train_ds = t.utils.data.TensorDataset(samples[train_idx], labels[train_idx])
        val_ds = t.utils.data.TensorDataset(samples[val_idx], labels[val_idx])
        train_loader = t.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = t.utils.data.DataLoader(val_ds, batch_size=len(val_idx), shuffle=True)

        # prepare the model
        model = models[split]
        if t.cuda.is_available():
            model = model.cuda()

        # prepare loss function and optimizer
        loss_function = t.nn.CrossEntropyLoss()
        optimizer = t.optim.Adam(model.parameters(), lr=lr)

        # train all epochs and evaluate
        for epoch in epochs:
            # train the model
            _train_epoch(model, train_loader, loss_function, optimizer)

            # collect current loss and accuracy
            loss[0, epoch] = _test_net(model, train_loader, loss_function)
            loss[1, epoch] = _test_net(model, val_loader, loss_function)
            accuracy[0, epoch] = _test_net(model, train_loader)
            accuracy[1, epoch] = _test_net(model, val_loader)

            if progress:
                pbar.update()

    # close the progress bar
    if progress:
        pbar.close()

    metrics = get_metrics_from_model(model, val_loader)
    return models, metrics


def train_subject_specific(subject, epochs=500, batch_size=32, lr=0.001, progress=True, plot=True):
    """
    Trains a subject specific model for the given subject

    Parameters:
     - subject: Integer in the Range 1 <= subject <= 9
     - epochs: Number of epochs to train
     - batch_size: Batch Size
     - lr: Learning Rate

    Returns: (model, metrics)
     - model:   t.nn.Module, trained model
     - metrics: t.tensor, size=[1, 4], accuracy, precision, recall, f1
    """
    # load the data
    train_samples, train_labels = get_data(subject, training=True)
    test_samples, test_labels = get_data(subject, training=False)
    train_loader = as_data_loader(train_samples, train_labels, batch_size=batch_size)
    test_loader = as_data_loader(test_samples, test_labels, batch_size=test_labels.shape[0])

    # prepare the model
    model = EEGNet(T=train_samples.shape[2])
    if t.cuda.is_available():
        model = model.cuda()

    # prepare loss function and optimizer
    loss_function = t.nn.CrossEntropyLoss()
    optimizer = t.optim.Adam(model.parameters(), lr=lr)

    # Early stopping is not allowed in this mode, because the testing data cannot be used for
    # training!
    return _train_net(subject, model, train_loader, test_loader, loss_function, optimizer,
                      epochs=epochs, early_stopping=False, progress=progress, plot=plot)


def _train_net(subject, model, train_loader, val_loader, loss_function, optimizer, scheduler=None,
               epochs=500, early_stopping=True, progress=True, plot=True):
    """
    Main training loop

    Parameters:
     - subject:        Integer, subject ID
     - model:          t.nn.Module (is set to training mode)
     - train_loader:   t.utils.data.DataLoader: training data
     - val_loader:     t.utils.data.DataLoader: validation data
     - loss_function:  function
     - optimizer:      t.optim.Optimizer
     - scheduler:      t.optim.lr_scheduler or None
     - epochs:         Integer, number of epochs to train
     - early_stopping: boolean, if True, store models for all epochs and select the one with the
                       highest validation accuracy
     - progress:       boolean, if True, show a progress bar (tqdm)
     - plot:           boolean, if True, generate all plots and store on disk

    Returns: (model, metrics)
     - model:   t.nn.Module, trained model
     - metrics: t.tensor, size=[1, 4], accuracy, precision, recall, f1

    Notes:
     - Model and data will not be moved to gpu, do this outside of this function.
     - When early_stopping is enabled, this function will store all intermediate models
    """

    # prepare result
    loss = t.zeros((2, epochs))
    accuracy = t.zeros((2, epochs))

    # prepare progress bar
    if progress:
        pbar = tqdm(total=epochs, desc=f"Subject {subject}, accuracy =    nan", ascii=True)

    # prepare early_stopping
    if early_stopping:
        early_stopping = EarlyStopping()

    # train model for all epochs
    for epoch in range(epochs):
        # train the model
        train_loss, train_accuracy = _train_epoch(model, train_loader, loss_function, optimizer,
                                                  scheduler=scheduler)

        # collect current loss and accuracy
        loss[0, epoch] = train_loss
        loss[1, epoch] = _test_net(model, val_loader, loss_function, train=False)
        accuracy[0, epoch] = train_accuracy
        accuracy[1, epoch] = _test_net(model, val_loader, train=False)

        # do early stopping
        if early_stopping:
            early_stopping.checkpoint(model, loss[1, epoch], accuracy[1, epoch], epoch)

        pbar.set_description(f"Subject {subject}, accuracy = {accuracy[1, epoch]:1.4f}",
                             refresh=False)
        pbar.update()

    # close the progress bar
    if progress:
        pbar.close()

    # get the best model
    if early_stopping:
        model, best_loss, best_accuracy, best_epoch = early_stopping.use_best_model(model)
        print(f"Early Stopping: using model at epoch {best_epoch+1} with accuracy {best_accuracy}")

    # generate plots
    if plot:
        generate_plots(subject, model, val_loader, loss, accuracy)

    metrics = get_metrics_from_model(model, val_loader)
    return model, metrics


def _train_epoch(model, loader, loss_function, optimizer, scheduler=None):
    """
    Trains a single epoch

    Parameters:
     - model:         t.nn.Module (is set to training mode)
     - loader:        t.utils.data.DataLoader
     - loss_function: function
     - optimizer:     t.optim.Optimizer
     - scheduler:     t.optim.lr_scheduler or None

    Returns: loss: float, accuracy: float
    """

    model.train(True)
    n_samples = 0
    running_loss = 0.0
    accuracy = 0.0
    for x, y in loader:
        # Forward step
        optimizer.zero_grad()
        output = model(x)
        loss = loss_function(output, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        # prepare loss and accuracy
        n_samples += x.shape[0]
        running_loss += loss * x.shape[0]
        decision = class_decision(output)
        accuracy += (decision == y).sum().item()

    running_loss = running_loss / n_samples
    accuracy = accuracy / n_samples
    return running_loss, accuracy


def _test_net(model, loader, loss_function=None, train=False):
    """
    Tests the model for accuracy

    Parameters:
     - model:         t.nn.Module (is set to testing mode)
     - loader:        t.utils.DataLoader
     - loss_function: function or None. If None, then accuracy is tested
     - train:         boolean, if the model is to be set into training or testing mode

    Returns: accuracy: float
    """
    # set the model into testing mode
    model.train(train)
    with t.no_grad():

        result = 0.0
        n_total = 0

        # get the data from the loader (only one batch will be available)
        for x, y in loader:

            # compute the output
            output = model(x)

            if loss_function is None:
                # compare the prediction
                yhat = output.argmax(dim=1)
                prediction_correct = yhat == y
                num_correct = prediction_correct.sum().item()
                n_total += x.shape[0]
                result += num_correct

            else:
                # compute the loss function
                loss = loss_function(output, y)
                result += loss
                n_total += 1

    return result / n_total
