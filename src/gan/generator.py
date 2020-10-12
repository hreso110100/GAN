from torch import nn, cat
from torch.nn import Upsample, Conv2d, Sequential, LeakyReLU, Dropout, Sigmoid, ZeroPad2d, ReLU, BatchNorm2d


class UNetDown(nn.Module):
    def __init__(self, input_size: int, output_filters: int, normalize=True):
        super(UNetDown, self).__init__()

        self.model = Sequential(
            Conv2d(input_size, output_filters, kernel_size=(4, 1), padding=(1, 0), stride=(2, 1), bias=False),
            LeakyReLU(0.2)
        )

        if normalize:
            self.model.add_module("BatchNorm2d", BatchNorm2d(output_filters, momentum=0.8))

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, input_size: int, output_filters: int, dropout=0.0):
        super(UNetUp, self).__init__()

        self.model = Sequential(
            Upsample(scale_factor=(2, 1)),
            ZeroPad2d((0, 0, 1, 0)),
            Conv2d(input_size, output_filters, kernel_size=(4, 1), stride=1, padding=(1, 0), bias=False),
            ReLU(inplace=True),
            BatchNorm2d(output_filters, momentum=0.8),
        )

        if dropout:
            self.model.add_module("Dropout", Dropout(dropout))

    def forward(self, layer, skip_input):
        layer = self.model(layer)
        layer = cat((layer, skip_input), 1)

        return layer


"""
Implementation based on UNet generator
"""


class Generator(nn.Module):
    def __init__(self, file_shape: tuple, output_filters=8, output_channels=2):
        super(Generator, self).__init__()

        # DownSampling
        self.down1 = UNetDown(file_shape[0], output_filters, normalize=False)
        self.down2 = UNetDown(output_filters, output_filters * 2)
        self.down3 = UNetDown(output_filters * 2, output_filters * 4)
        self.down4 = UNetDown(output_filters * 4, output_filters * 8)
        self.down5 = UNetDown(output_filters * 8, output_filters * 8)
        self.down6 = UNetDown(output_filters * 8, output_filters * 8)
        self.down7 = UNetDown(output_filters * 8, output_filters * 8)
        self.down8 = UNetDown(output_filters * 8, output_filters * 8)

        # UpSampling
        self.up1 = UNetUp(output_filters * 8, output_filters * 8)
        self.up2 = UNetUp(output_filters * 16, output_filters * 8)
        self.up3 = UNetUp(output_filters * 16, output_filters * 8)
        self.up4 = UNetUp(output_filters * 16, output_filters * 8)
        self.up5 = UNetUp(output_filters * 16, output_filters * 4)
        self.up6 = UNetUp(output_filters * 8, output_filters * 2)
        self.up7 = UNetUp(output_filters * 4, output_filters)

        self.last = nn.Sequential(
            Upsample(scale_factor=(2, 4)),
            ZeroPad2d((0, 0, 1, 0)),
            Conv2d(output_filters * 2, output_channels, kernel_size=(4, 4), stride=1, padding=(1, 0)),
            Sigmoid(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.last(u7)
