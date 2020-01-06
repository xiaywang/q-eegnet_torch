import numpy as np
import torch as t
import torch.quantization as tq
import torch.nn.functional as F

# set the backend engine
t.backends.quantized.engine = 'qnnpack'


class EEGNetQuant(t.nn.Module):
    """
    EEGNet
    """
    def __init__(self, qconfig, F1=8, D=2, F2=None, C=22, T=1125, N=4, p_dropout=0.5,
                 dropout_type='TimeDropout2D'):
        """
        qconfig:      Quantization Configuration
        F1:           Number of spectral filters
        D:            Number of spacial filters (per spectral filter), F2 = F1 * D
        F2:           Number or None. If None, then F2 = F1 * D
        C:            Number of EEG channels
        T:            Number of time samples
        N:            Number of classes
        p_dropout:    Dropout Probability
        dropout_type: string, either 'dropout', 'SpatialDropout2d' or 'TimeDropout2D'
        """
        super(EEGNetQuant, self).__init__()

        # prepare network constants
        if F2 is None:
            F2 = F1 * D

        # Prepare Dropout Type
        if dropout_type.lower() == 'dropout':
            dropout = t.nn.Dropout
        elif dropout_type.lower() == 'spatialdropout2d':
            dropout = t.nn.Dropout2d
        elif dropout_type.lower() == 'timedropout2d':
            dropout = TimeDropout2d
        else:
            raise ValueError("dropout_type must be one of SpatialDropout2d, Dropout or "
                             "WrongDropout2d")

        # store local values
        self.F1, self.D, self.F2, self.C, self.T, self.N = (F1, D, F2, C, T, N)
        self.p_dropout, self.dropout_type = (p_dropout, dropout_type)
        self.qconfig = qconfig

        # Number of input neurons to the final fully connected layer
        n_features = (((T - 1) // 8) - 1) // 8

        # input quantization
        self.quant = tq.QuantStub(qconfig)

        # Block 1
        # self.conv1_pad = t.nn.ZeroPad2d((31, 32, 0, 0))
        self.conv1_bn = t.nn.intrinsic.qat.ConvBn2d(1, F1, (1, 64), padding=(0, 31), momentum=0.01,
                                                    qconfig=qconfig)
        self.conv2_bn_relu = t.nn.intrinsic.qat.ConvBnReLU2d(F1, D * F1, (C, 1), groups=F1,
                                                             momentum=0.01, qconfig=qconfig)
        self.pool1 = t.nn.AvgPool2d((1, 8))
        self.dropout1 = t.nn.Dropout(p=p_dropout)

        # Block 2
        # Separable Convolution (as described in the paper) is a depthwise convolution followed by
        # a pointwise convolution.
        # self.sep_conv_pad = t.nn.ZeroPad2d((7, 8, 0, 0))
        # TODO we might want to add batch norm between both convolutions
        self.sep_conv1 = t.nn.qat.Conv2d(D * F1, D * F1, (1, 16), groups=D * F1, bias=False,
                                         padding=(0, 7), qconfig=qconfig)
        self.sep_conv2_bn_relu = t.nn.intrinsic.qat.ConvBnReLU2d(D * F1, F2, (1, 1), momentum=0.01,
                                                                 qconfig=qconfig)
        self.pool2 = t.nn.AvgPool2d((1, 8))
        self.dropout2 = dropout(p=p_dropout)

        self.flatten = t.nn.Flatten()
        self.fc = t.nn.qat.Linear(F2 * n_features, N, bias=True, qconfig=qconfig)

        self.dequant = tq.DeQuantStub()

    def forward(self, x):

        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])

        x = self.quant(x)

        # Block 1
        # x = self.conv1_pad(x)
        x = self.conv1_bn(x)
        x = self.conv2_bn_relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Block2
        # x = self.sep_conv_pad(x)
        x = self.sep_conv1(x)
        x = self.sep_conv2_bn_relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # Classification
        x = self.flatten(x)
        x = self.fc(x)

        x = self.dequant(x)

        return x

    def initialize_params(self, weight_init=t.nn.init.xavier_uniform_, bias_init=t.nn.init.zeros_,
                          weight_gain=None, bias_gain=None):
        """
        Initializes all the parameters of the model

        Parameters:
         - weight_init: t.nn.init inplace function
         - bias_init:   t.nn.init inplace function
         - weight_gain: float, if None, don't use gain for weights
         - bias_gain:   float, if None, don't use gain for bias

        """
        # use gain only if xavier_uniform or xavier_normal is used
        weight_params = {}
        bias_params = {}
        if weight_gain is not None:
            weight_params['gain'] = weight_gain
        if bias_gain is not None:
            bias_params['gain'] = bias_gain

        def init_weight(m):
            if isinstance(m, t.nn.Conv2d) or isinstance(m, t.nn.Linear):
                weight_init(m.weight, **weight_params)
            if isinstance(m, t.nn.Linear):
                bias_init(m.bias, **bias_params)

        self.apply(init_weight)

    def is_cuda(self):
        is_cuda = False
        for param in self.parameters():
            if param.is_cuda:
                is_cuda = True
        return is_cuda


class TimeDropout2d(t.nn.Dropout2d):
    """
    Dropout layer, where the last dimension is treated as channels
    """
    def __init__(self, p=0.5, inplace=False):
        """
        See t.nn.Dropout2d for parameters
        """
        super(TimeDropout2d, self).__init__(p=p, inplace=inplace)

    def forward(self, input):
        if self.training:
            input = input.permute(0, 3, 1, 2)
            input = F.dropout2d(input, self.p, True, self.inplace)
            input = input.permute(0, 2, 3, 1)
        return input
