import torch.nn as nn
from collections import OrderedDict

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    ("conv", nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)),
                                    ("batch_norm", nn.BatchNorm2d(out_channels)),
                                    ("relu", nn.ReLU()),
                                ]
                            )
                        ),
                    ),
                    (
                        "conv2",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    ("conv", nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)),
                                    ("batch_norm", nn.BatchNorm2d(out_channels)),
                                ]
                            )
                        ),
                    ),
                ]
            )
        )
        
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                OrderedDict(
                    [
                        ("downsample_conv", nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)),
                        ("downsample_batch_norm", nn.BatchNorm2d(out_channels)),
                    ]
                )
            )
        else:
            self.downsample = None
        self.relu = nn.ReLU()

    def forward(self, x):
        # F(x)
        F = self.conv_block(x)
        if self.downsample:
            residual = self.downsample(x)
        else:
            residual = x
        
        # IMPORTANT BIT: we sum the result of the
        # convolutions to the input image
        H = F + residual
        # Now we apply ReLU and return
        return self.relu(H)

def get_conv_layer(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, pool=False):
        conv_block = nn.Sequential(
            OrderedDict(
                [
                ("conv", nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)),
                ("batch_norm", nn.BatchNorm2d(out_channels)),
                ("relu", nn.ReLU()),
                ]
            )
        )
        conv_layer = OrderedDict([("conv_block", conv_block)])
        
        if pool:
            conv_layer["max_pool"] = nn.MaxPool2d(kernel_size = 2, stride = 2)
        return nn.Sequential(conv_layer)

def get_fc_layer(in_channels, out_channels, dropout):
    sublayers = OrderedDict(
        [
            ("dropout", nn.Dropout(dropout)),
            ("linear", nn.Linear(in_channels, out_channels)),
            ("relu", nn.ReLU()),
        ]
    )
    return nn.Sequential(sublayers)