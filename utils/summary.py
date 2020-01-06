"""
torchsummary package: https://github.com/sksq96/pytorch-summary

Edited by Tibor Schneider
"""

import torch
import torch.nn as nn

from collections import OrderedDict
import numpy as np

from tabulate import tabulate

_SUMMARY_ALREADY_SHOWN = False
_IGNORE_LAYERS = [torch.quantization.Observer]


def print_summary(model, optimizer, loss_function, scheduler=None, batch_size=-1, device="cuda",
                  print_mem=False, print_once=True):
    """
    Print entire summary of the training.

    Shows the model (including hyperparameters) and optimizer (including loss and scheduler)
    """
    global _SUMMARY_ALREADY_SHOWN
    if not print_once or not _SUMMARY_ALREADY_SHOWN:
        _SUMMARY_ALREADY_SHOWN = True
        print("")

        # print optimizer, loss function and scheduler
        optim_table = []
        if optimizer is None:
            optim_table.append(["Optimizer", "None"])
        elif isinstance(optimizer, torch.optim.Adam):
            optim_table.append(["Optimizer", "Adam"])
            optim_table.append(["> learning rate", optimizer.defaults['lr']])
        elif isinstance(optimizer, torch.optim.SGD):
            optim_table.append(["Optimizer", "SGD"])
            optim_table.append(["> learning rate", optimizer.defaults['lr']])
        else:
            raise NotImplementedError(f"{type(optimizer)} is not implemented!")

        optim_table.append(["Loss function", str(loss_function)])

        if scheduler is None:
            optim_table.append(["Scheduler", "None"])
        elif isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            optim_table.append(["Scheduler", "Reduce LR on platau"])
            optim_table.append(["> factor", scheduler.factor])
            optim_table.append(["> patience", scheduler.patience])
            optim_table.append(["> threshold", scheduler.threshold])
        else:
            raise NotImplementedError(f"{type(scheduler)} is not implemented!")
        print(tabulate(optim_table, ["Learning Method", ""], tablefmt='orgtbl'))
        print("")

        # print hyper parameters
        param_table = [["F1 (# spectral filters)", model.F1],
                       ["F2 (# spatial filters)", model.F2],
                       ["Dropout probability", model.p_dropout],
                       ["Dropout type", model.dropout_type],
                       ["Constrain weights", model.__dict__.get('constrain_w', False)],
                       ["Activation type", 'ELU' if model.__dict__.get('activation', 'relu') ==
                        'elu' else 'ReLU']]
        print(tabulate(param_table, ['Hyperparameter', 'Value'], tablefmt='orgtbl'))
        print("")

        # print the model summary
        _model_summary(model, (model.C, model.T), batch_size=1, device=device, print_mem=print_mem)


def _model_summary(model, input_size, batch_size=-1, device="cuda", print_mem=False):

    def register_hook(module):

        def hook(module, input, output):
            if any([isinstance(module, layer) for layer in _IGNORE_LAYERS]):
                return
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            summary[m_key]["params"] = []
            for parameter in module.parameters():
                params = torch.prod(torch.LongTensor(list(parameter.size())))
                summary[m_key]["params"].append((params, parameter.requires_grad))

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    # prepare network table
    net_table = []
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        layer_nb_params = sum([nb for nb, _ in summary[layer]["params"]])
        net_table.append([layer, summary[layer]["output_shape"], layer_nb_params])
        total_output += np.prod(summary[layer]["output_shape"])
        for param in summary[layer]["params"]:
            nb_params, is_trainable = param
            total_params += nb_params
            if is_trainable:
                trainable_params += nb_params
    print(tabulate(net_table, ["layer (type)", "Output shape", "# params"], tablefmt='orgtbl'))

    print("")
    # prepare number of param table
    num_table = [["Total params", total_params],
                 ["Trainable params", trainable_params],
                 ["Non-trainable params", total_params - trainable_params]]
    print(tabulate(num_table, tablefmt='orgtbl'))
    print("")
    if print_mem:
        # assume 4 bytes/number (float on cuda).
        total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
        total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
        total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
        total_size = total_params_size + total_output_size + total_input_size

        mem_table = [["Input size (MB)", total_input_size],
                     ["Forward/backward pass size (MB)", total_output_size],
                     ["Param size (MB)", total_params_size],
                     ["Estimated Total Size (MB)", total_size]]
        print(tabulate(mem_table, tablefmt='orgtbl'))
        print("")
