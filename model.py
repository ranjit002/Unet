import torch
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        This layer applies the following steps twice:
        1)A convolution with no biases,
        2)A normalisation (mean, standard deviation) to the convoluted tensor
        3)ReLU

        Args:
            in_channels (int): number of channels inputted into the network
            out_channels (int): number of channels to be outputted by the convolution
        """
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(  # 1st conv
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(  # 2nd conv
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)


class UNET(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        layer_sizes=[64, 128, 256],
    ):
        """
        Args:
            in_channels (int): number of channels inputted into the model
            out_channels (int): number of channels to be outputted by the model
            layer_sizes (list): sizes of the channels in the encoder layer
        """
        super().__init__()

        # Check channel sizes are valid
        for idx in range(len(layer_sizes) - 1):
            if 2 * layer_sizes[idx] != layer_sizes[idx + 1]:
                raise ValueError(
                    "The decoder channel sizes are invalid. Must increase in factors of 2."
                )

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        # Build Encoder
        for size in layer_sizes:
            self.encoders.append(DoubleConv(in_channels, size))
            in_channels = size

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = DoubleConv(layer_sizes[-1], layer_sizes[-1] * 2)

        # Build Decoder
        for size in reversed(layer_sizes):
            self.decoders.append(
                nn.ConvTranspose2d(size * 2, size, kernel_size=2, padding=2, stride=1)
            )
            self.decoders.append(DoubleConv(size * 2, size))

        self.final_conv = nn.Conv2d(
            layer_sizes[0], out_channels, kernel_size=1, padding=1, stride=1
        )

    def forward(self, x):
        skip_connections = []

        for layer in self.encoders:
            x = layer(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for idx in range(0, len(self.decoders), 2):
            x = self.decoders[idx](x)  # Up convolution

            concat_skip = torch.cat(  # Skip connection
                (skip_connections[idx // 2], x), dim=1
            )
            x = self.decoders[idx + 1](concat_skip)

        return self.final_conv(x)
