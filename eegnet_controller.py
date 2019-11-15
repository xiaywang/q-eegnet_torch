import torch as t

from tqdm import tqdm

from utils.get_data import as_data_loader


def _train_net(model, samples, labels, epochs=500, batch_size=32, lr=0.001):

    # generate dataLoader
    loader = as_data_loader(samples, labels, batch_size=batch_size)

    # move model to cuda
    if t.cuda.is_available():
        model = model.cuda()

    # set the model into training mode
    model.train(True)

    # prepare optimizer
    loss_function = t.nn.CrossEntropyLoss()
    optimizer = t.optim.Adam(model.parameters(), lr=lr)

    with tqdm(total=epochs, desc="training model") as pbar:

        for epoch in range(epochs):
            running_loss = 0.0

            for x, y in loader:
                optimizer.zero_grad()
                output = model(x)
                loss = loss_function(output, y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            pbar.set_description(f"training model: loss = {running_loss / len(loader):1.4f}")
            pbar.update(1)
            running_loss = 0.0


def _test_net(model, samples, labels):
    # generate loader
    n_test_samples = labels.shape[0]
    loader = as_data_loader(samples, labels, batch_size=n_test_samples)

    # move model to cuda (probably already done)
    if t.cuda.is_available():
        model = model.cuda()

    # set the model into testing mode
    model.train(False)

    # get the data from the loader (only one batch will be available)
    x, y = iter(loader).next()

    output = model(x)
    yhat = output.argmax(dim=1)
    prediction_correct = yhat == y
    num_correct = prediction_correct.sum().item()
    print(f"accuracy: {num_correct / n_test_samples}")
    return num_correct / n_test_samples
