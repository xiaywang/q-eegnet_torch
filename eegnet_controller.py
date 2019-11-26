import torch as t

from tqdm import tqdm

from eegnet import EEGNet
from utils.get_data import get_data, as_data_loader, as_tensor
from utils.kfold_cv import KFoldCV


def train_subject_specific_cv(subject, n_splits=4, epochs=500, batch_size=32, lr=0.001,
                              progress=True):
    """
    Trains a subject specific model for the given subject, using K-Fold Cross Validation

    Parameters:
     - subject: Integer in the Range 1 <= subject <= 9
     - n_splits: Number of splits for K-Fold CV
     - epochs: Number of epochs to train
     - batch_size: Batch Size
     - lr: Learning Rate
     - progress: bool, if True, displays a progress bar

    Returns: (models, loss, accuracy)
     - models:   List of t.nn.Module, size = [n_splits]
     - loss:     t.tensor, size = [2, epochs], mean loss of all CV splits
     - accuracy: t.tensor, size = [2, epochs], mean accuracy of all CV splits
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

    return models, loss, accuracy


def train_subject_specific(subject, epochs=500, batch_size=32, lr=0.001, progress=True):
    """
    Trains a subject specific model for the given subject

    Parameters:
     - subject: Integer in the Range 1 <= subject <= 9
     - epochs: Number of epochs to train
     - batch_size: Batch Size
     - lr: Learning Rate

    Returns: (model, loss, accuracy)
     - model:    t.nn.Module
     - loss:     t.tensor, size = [2, epochs]
     - accuracy: t.tensor, size = [2, epochs]
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
    # scheduler = t.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

    # prepare result
    loss = t.zeros((2, epochs))
    accuracy = t.zeros((2, epochs))

    # prepare progress bar
    if progress:
        pbar = tqdm(total=epochs, desc=f"Subject {subject}, accuracy =    nan", ascii=True)

    # train model for all epochs
    for epoch in range(epochs):
        # train the model
        _train_epoch(model, train_loader, loss_function, optimizer)

        # collect current loss and accuracy
        loss[0, epoch] = _test_net(model, train_loader, loss_function)
        loss[1, epoch] = _test_net(model, test_loader, loss_function)
        accuracy[0, epoch] = _test_net(model, train_loader)
        accuracy[1, epoch] = _test_net(model, test_loader)

        pbar.set_description(f"Subject {subject}, accuracy = {accuracy[1, epoch]:1.4f}",
                             refresh=False)
        pbar.update()

    # close the progress bar
    if progress:
        pbar.close()

    return model, loss, accuracy


def _train_epoch(model, loader, loss_function, optimizer, scheduler=None):
    """
    Trains a single epoch

    Parameters:
     - model:         t.nn.Module (is set to training mode)
     - loader:        t.utils.data.DataLoader
     - loss_function: function
     - optimizer:     t.optim.Optimizer
     - scheduler:     t.optim.lr_scheduler or None

    Returns: loss: float
    """

    model.train(True)
    running_loss = 0.0
    for x, y in loader:
        optimizer.zero_grad()
        output = model(x)
        loss = loss_function(output, y)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        running_loss += loss
    return loss


def _test_net(model, loader, loss_function=None):
    """
    Tests the model for accuracy

    Parameters:
     - model:         t.nn.Module (is set to testing mode)
     - loader:        t.utils.DataLoader
     - loss_function: function or None. If None, then accuracy is tested

    Returns: accuracy: float
    """
    # set the model into testing mode
    model.train(False)
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
