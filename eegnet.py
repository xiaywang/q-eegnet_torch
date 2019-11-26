import torch as t
import torch.nn.functional as F


class EEGNet(t.nn.Module):
    """
    EEGNet
    """
    def __init__(self, F1=8, D=2, F2=None, C=22, T=1750, N=4, p_dropout=0.5):
        """
        F1: Number of spectral filters
        D: Number of spacial filters (per spectral filter), F2 = F1 * D
        C: Number of EEG channels
        T: Number of time samples
        N: Number of classes
        p_dropout
        """
        super(EEGNet, self).__init__()

        # prepare network constants
        if F2 is None:
            F2 = F1 * D

        # Number of input neurons to the final fully connected layer
        # TODO the (-1) comes from the fact we don't ues correct same padding.
        n_features = (((T - 1) // 8) - 1) // 8

        # Block 1
        # TODO this convolution should have padding mode=same, but this is not possible for an even
        #      kernel size. Therefore, we have exactly one output feature less than the net
        #      described in the paper
        self.conv1 = t.nn.Conv2d(1, F1, (1, 64), padding=(0, 31))
        self.batch_norm1 = t.nn.BatchNorm2d(F1)
        self.conv2 = t.nn.Conv2d(F1, D * F1, (C, 1), groups=F1) # depthwise convolution
        self.batch_norm2 = t.nn.BatchNorm2d(D * F1)
        self.dropout1 = t.nn.Dropout(p=p_dropout)

        # Block 2
        # Separable Convolution (as described in the paper) is a depthwise convolution followed by
        # a pointwise convolution.
        # TODO padding should be same, but we loose one sample
        self.sep_conv1 = t.nn.Conv2d(D * F1, D * F1, (1, 16), groups=D * F1, padding=(0, 7))
        self.sep_conv2 = t.nn.Conv2d(D * F1, F2, (1, 1)) # pointwise conv
        self.batch_norm3 = t.nn.BatchNorm2d(F2)
        self.dropout2 = t.nn.Dropout(p=p_dropout)

        # Fully connected layer (classifier)
        self.fc = t.nn.Linear(F2 * n_features, N, bias=False)

        # initialize all weights with xavier_normal (same as keras)
        def init_weight(m):
            if isinstance(m, t.nn.Conv2d) or isinstance(m, t.nn.Linear):
                t.nn.init.xavier_normal_(m.weight)
        self.apply(init_weight)

    def forward(self, x):
        # reshape vector from (s, C, T) to (s, 1, C, T)
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])

        # input dimensions: (s, 1, C, T)

        # Block 1
        x = self.conv1(x)            # output dim: (s, F1, C, T-1)
        x = self.batch_norm1(x)
        x = self.conv2(x)            # output dim: (s, D * F1, 1, T-1)
        x = self.batch_norm2(x)
        x = F.elu(x)
        x = F.avg_pool2d(x, (1, 8))  # output dim: (s, D * F1, 1, T // 8)
        x = self.dropout1(x)

        # Block2
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
