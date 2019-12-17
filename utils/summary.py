"""
torchsummary package: https://github.com/sksq96/pytorch-summary

Edited by Tibor Schneider
"""

import torch
import torch.nn as nn

from collections import OrderedDict
import numpy as np

from tabulate import tabulate


def summary(model, input_size, batch_size=-1, device="cuda", print_mem=False, model_params=None):

    def register_hook(module):

        def hook(module, input, output):
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

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

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

    # print all model parameters
    if model_params:
        print("")
        param_table = [[key, value] for key, value in model_params.items()]
        print(tabulate(param_table, ['Hyperparameter', 'Value'], tablefmt='orgtbl'))

    print("")

    # prepare network table
    net_table = []
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        net_table.append([layer, summary[layer]["output_shape"], summary[layer]["nb_params"]])
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"]:
                trainable_params += summary[layer]["nb_params"]
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
    # return summary
