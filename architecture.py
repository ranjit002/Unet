from torch import cat, nn


class BCELogitsWeight(nn.Module):
    def __init__(self, alpha=0.02):
        """
        Binary Cross Entropy, with a rescaling of the cross entropies
        so that the two classes are proportionally represented.

        Args:
            alpha (float): Fraction of classes represented as = 1 in the target mask
        """
        super().__init__()
        self.alpha = alpha

    def forward(self, output, target):
        """
        Args:
            output (tensor): prediction of model WITHOUT sigmoid applied (Logits)
            target (tensor): desired output of model AFTER sigmoid (mask)
        """
        assert output.shape == target.shape, "Prediction and target shape mismatch"

        pred = output.sigmoid().view(-1)  # Apply sigmoid and flatten tensor
        target_flat = target.view(-1)

        epsilon = 1e-6
        pred = pred.clamp(
            epsilon, 1 - epsilon
        )  # Prevent log(pred) and log(1 - pred) from overflowing

        return -(
            (1.0 - self.alpha) * target_flat * pred.log()
            + self.alpha * (1.0 - target_flat) * (1.0 - pred).log()
        ).mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-7):
        """
        Dice loss with smoothing

        Args:
            smooth (float): smoothing to prevent division by zero in dice coeffn denominator
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, output, target):
        """
        Args:
            output (tensor): prediction of model BEFORE sigmoid is applied
            target (tensor): desired output of the model AFTER sigmoid is applied (i.e the mask)
        """
        assert output.shape == target.shape, "Prediction and target shape mismatch"

        pred = output.sigmoid().view(-1)  # Apply sigmoid and flatten tensor
        target_flat = target.view(-1)

        intersection = (pred * target_flat).sum()
        union = pred.sum() + target_flat.sum()

        return 1 - (2.0 * intersection + self.smooth) / (union + self.smooth)


class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
    ):
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

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, 3, 1, 1, bias=False
            ),  # First convolutional layer
            # NOTE: Bias=False makes it so the model learns no biases at the convolutions
            nn.BatchNorm2d(
                out_channels
            ),  # Resize tensor to new mean and standard deviation (which are learned parameters)
            nn.ReLU(inplace=True),
            # Second layer
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        features=[64, 128, 256],
    ):
        """
        Args:
            in_channels (int): number of channels inputted into the model
            out_channels (int): number of channels to be outputted by the model
            features (list):
        """
        super().__init__()

        # Check encoder sizes are valid.
        for idx in range(len(features) - 1):
            if 2 * features[idx] != features[idx + 1]:
                raise ValueError(
                    "The decoder layer sizes are invalid. Must increase in factors of 2."
                )

        self.ups = nn.ModuleList()  # Store encoding layers
        self.downs = nn.ModuleList()  # Store decoding layers

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Maxpooling layer

        # Encoder
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))  # Multiple DoubleConvs
            in_channels = feature

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(  # Up convolution
                    feature * 2,
                    feature,
                    kernel_size=2,
                    stride=2,
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[
            ::-1
        ]  # Reverse skip connections to feed into decoder

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # Up convolution
            skip_connection = skip_connections[idx // 2]

            concat_skip = cat(
                (skip_connection, x), dim=1
            )  # Concatenate tensor with skip connection
            x = self.ups[idx + 1](concat_skip)  # DoubleConv

        return self.final_conv(x)
