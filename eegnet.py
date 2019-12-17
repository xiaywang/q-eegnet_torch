import torch as t
import torch.nn.functional as F


class EEGNet(t.nn.Module):
    """
    EEGNet
    """
    def __init__(self, F1=8, D=2, F2=None, C=22, T=1125, N=4, p_dropout=0.5, reg_rate=0.25,
                 activation='relu', constrain_w=False, dropout_type='dropout'):
        """
        F1:           Number of spectral filters
        D:            Number of spacial filters (per spectral filter), F2 = F1 * D
        F2:           Number or None. If None, then F2 = F1 * D
        C:            Number of EEG channels
        T:            Number of time samples
        N:            Number of classes
        p_dropout:    Dropout Probability
        reg_rate:     Regularization (L1) of the final linear layer (fc)
                      This parameter is ignored when constrain_w is not asserted
        activation:   string, either 'elu' or 'relu'
        constrain_w:  bool, if True, constrain weights of spatial convolution and final fc-layer
        dropout_type: string, either 'dropout' or 'SpatialDropout2d'
        """
        super(EEGNet, self).__init__()

        # prepare network constants
        if F2 is None:
            F2 = F1 * D

        # check the activation input
        activation = activation.lower()
        assert activation in ['elu', 'relu']

        # Prepare Dropout Type
        if dropout_type.lower() == 'dropout':
            dropout = t.nn.Dropout
        elif dropout_type.lower() == 'spatialdropout2d':
            dropout = t.nn.Dropout2d
        elif dropout_type.lower() == 'wrongdropout2d':
            dropout = WrongDropout2d
        else:
            raise ValueError("dropout_type must be one of SpatialDropout2d, Dropout or "
                             "WrongDropout2d")

        # store local values
        self.F1, self.D, self.F2, self.C, self.T, self.N = (F1, D, F2, C, T, N)
        self.p_dropout, self.reg_rate, self.activation = (p_dropout, reg_rate, activation)
        self.constrain_w, self.dropout_type = (constrain_w, dropout_type)

        # Number of input neurons to the final fully connected layer
        n_features = (T // 8) // 8

        # Block 1
        self.conv1_pad = t.nn.ZeroPad2d((32, 31, 0, 0))
        self.conv1 = t.nn.Conv2d(1, F1, (1, 64), bias=False)
        self.batch_norm1 = t.nn.BatchNorm2d(F1, momentum=0.01, eps=0.001)
        # By setting groups=F1 (input dimension), we get a depthwise convolution
        if constrain_w:
            self.conv2 = ConstrainedConv2d(F1, D * F1, (C, 1), groups=F1, bias=False,
                                           max_weight=1.0)
        else:
            self.conv2 = t.nn.Conv2d(F1, D * F1, (C, 1), groups=F1, bias=False)
        self.batch_norm2 = t.nn.BatchNorm2d(D * F1, momentum=0.01, eps=0.001)
        self.activation1 = t.nn.ELU() if activation == 'elu' else t.nn.ReLU()
        self.dropout1 = dropout(p=p_dropout)

        # Block 2
        # Separable Convolution (as described in the paper) is a depthwise convolution followed by
        # a pointwise convolution.
        self.sep_conv_pad = t.nn.ZeroPad2d((8, 7, 0, 0))
        self.sep_conv1 = t.nn.Conv2d(D * F1, D * F1, (1, 16), groups=D * F1, bias=False)
        self.sep_conv2 = t.nn.Conv2d(D * F1, F2, (1, 1), bias=False)
        self.batch_norm3 = t.nn.BatchNorm2d(F2, momentum=0.01, eps=0.001)
        self.activation2 = t.nn.ELU() if activation == 'elu' else t.nn.ReLU()
        self.dropout2 = dropout(p=p_dropout)

        # Fully connected layer (classifier)
        if constrain_w:
            self.fc = ConstrainedLinear(F2 * n_features, N, bias=True, max_weight=reg_rate)
        else:
            self.fc = t.nn.Linear(F2 * n_features, N, bias=True)

    def forward(self, x):

        # reshape vector from (s, C, T) to (s, 1, C, T)
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])

        # input dimensions: (s, 1, C, T)

        # Block 1
        x = self.conv1_pad(x)
        x = self.conv1(x)            # output dim: (s, F1, C, T-1)
        x = self.batch_norm1(x)
        x = self.conv2(x)            # output dim: (s, D * F1, 1, T-1)
        x = self.batch_norm2(x)
        x = self.activation1(x)
        x = F.avg_pool2d(x, (1, 8))  # output dim: (s, D * F1, 1, T // 8)
        x = self.dropout1(x)

        # Block2
        x = self.sep_conv_pad(x)
        x = self.sep_conv1(x)        # output dim: (s, D * F1, 1, T // 8 - 1)
        x = self.sep_conv2(x)        # output dim: (s, F2, 1, T // 8 - 1)
        x = self.batch_norm3(x)
        x = self.activation2(x)
        x = F.avg_pool2d(x, (1, 8))  # output dim: (s, F2, 1, T // 64)
        x = self.dropout2(x)

        # Classification
        # flatten all the dimensions
        x = x.view(x.shape[0], -1)   # output dim: (s, F2 * (T // 64))
        x = self.fc(x)               # output dim: (s, N)

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

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class ConstrainedConv2d(t.nn.Conv2d):
    """
    Regularized Convolution, where the weights are clamped between two values.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, padding_mode='zeros', max_weight=1.0):
        """
        See t.nn.Conv2d for parameters.

        Parameters:
         - max_weight: float, all weights are clamped between -max_weight and max_weight
        """
        super(ConstrainedConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                                kernel_size=kernel_size, stride=stride,
                                                padding=padding, dilation=dilation, groups=groups,
                                                bias=bias, padding_mode=padding_mode)
        self.max_weight = max_weight

    def forward(self, input):
        return self.conv2d_forward(input, self.weight.clamp(min=-self.max_weight,
                                                            max=self.max_weight))


class ConstrainedLinear(t.nn.Linear):
    """
    Regularized Linear Transformation, where the weights are clamped between two values
    """

    def __init__(self, in_features, out_features, bias=True, max_weight=1.0):
        """
        See t.nn.Linear for parameters

        Parameters:
         - max_weight: float, all weights are clamped between -max_weight and max_weight
        """
        super(ConstrainedLinear, self).__init__(in_features=in_features, out_features=out_features,
                                                bias=bias)
        self.max_weight = max_weight

    def forward(self, input):
        return F.linear(input, self.weight.clamp(min=-self.max_weight, max=self.max_weight),
                        self.bias)


class WrongDropout2d(t.nn.Dropout2d):
    """
    Dropout layer, where the last dimension is treated as channels
    """
    def __init__(self, p=0.5, inplace=False):
        """
        See t.nn.Dropout2d for parameters
        """
        super(WrongDropout2d, self).__init__(p=p, inplace=inplace)

    def forward(self, input):
        if self.training:
            input = input.permute(0, 3, 1, 2)
            input = F.dropout2d(input, self.p, True, self.inplace)
            input = input.permute(0, 2, 3, 1)
        return input
