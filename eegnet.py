import torch as t
import torch.nn.functional as F
import traceback
import sys


class EEGNet(t.nn.Module):
    """
    EEGNet
    """
    def __init__(self, F1=8, D=2, F2=None, C=22, T=1125, N=4, p_dropout=0.5, reg_rate=0.25):
        """
        F1: Number of spectral filters
        D: Number of spacial filters (per spectral filter), F2 = F1 * D
        F2: Number or None. If None, then F2 = F1 * D
        C: Number of EEG channels
        T: Number of time samples
        N: Number of classes
        p_dropout: Dropout Probability
        reg_rate: Regularization (L1) of the final linear layer (fc)
        """
        super(EEGNet, self).__init__()

        # prepare network constants
        if F2 is None:
            F2 = F1 * D

        # Number of input neurons to the final fully connected layer
        n_features = (T // 8) // 8

        # Block 1
        self.conv1_pad = t.nn.ZeroPad2d((32, 31, 0, 0))
        self.conv1 = t.nn.Conv2d(1, F1, (1, 64), bias=False)
        self.batch_norm1 = t.nn.BatchNorm2d(F1)
        # By setting groups=F1 (input dimension), we get a depthwise convolution
        self.conv2 = ConstrainedConv2d(F1, D * F1, (C, 1), groups=F1, bias=False, max_weight=1.0)
        self.batch_norm2 = t.nn.BatchNorm2d(D * F1)
        self.dropout1 = t.nn.Dropout(p=p_dropout)

        # Block 2
        # Separable Convolution (as described in the paper) is a depthwise convolution followed by
        # a pointwise convolution.
        self.sep_conv_pad = t.nn.ZeroPad2d((8, 7, 0, 0))
        self.sep_conv1 = t.nn.Conv2d(D * F1, D * F1, (1, 16), groups=D * F1, bias=False)
        self.sep_conv2 = t.nn.Conv2d(D * F1, F2, (1, 1), bias=False)
        self.batch_norm3 = t.nn.BatchNorm2d(F2)
        self.dropout2 = t.nn.Dropout(p=p_dropout)

        # Fully connected layer (classifier)
        self.fc = ConstrainedLinear(F2 * n_features, N, bias=True, max_weight=reg_rate)

        # initialize all weights with xavier_normal, except the bias, which must be initialized
        # with 0. This is the same as the default initialization for keras
        def init_weight(m):
            if isinstance(m, t.nn.Conv2d) or isinstance(m, t.nn.Linear):
                t.nn.init.xavier_normal_(m.weight)
            if isinstance(m, t.nn.Linear):
                m.bias.data.zero_()
        self.apply(init_weight)

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
        x = F.elu(x)
        x = F.avg_pool2d(x, (1, 8))  # output dim: (s, D * F1, 1, T // 8)
        x = self.dropout1(x)

        # Block2
        x = self.sep_conv_pad(x)
        x = self.sep_conv1(x)        # output dim: (s, D * F1, 1, T // 8 - 1)
        x = self.sep_conv2(x)        # output dim: (s, F2, 1, T // 8 - 1)
        x = self.batch_norm3(x)
        x = F.elu(x)
        x = F.avg_pool2d(x, (1, 8))  # output dim: (s, F2, 1, T // 64)
        x = self.dropout2(x)

        # flatten all the dimensions
        x = x.view(x.shape[0], -1)   # output dim: (s, F2 * (T // 64))

        # Classification
        x = self.fc(x)               # output dim: (s, N)
        # x = F.softmax(x, dim=0) # softmax will be applied in the loss function

        return x

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
