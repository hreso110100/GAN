from torch import nn, cat
from torch.nn import Upsample, Conv2d, Sequential, LeakyReLU, BatchNorm2d, Dropout, ReLU, Sigmoid

"""
    Implementation based on UNet generator
"""


class Generator(nn.Module):

    def __init__(self, file_shape: tuple, filters: int):
        super(Generator, self).__init__()
        self.file_shape = file_shape
        self.filters = filters

        # Downsampling
        self.down_1 = self.conv2d(self.filters, batch_norm=False)
        self.down_2 = self.conv2d(self.filters * 2)
        self.down_3 = self.conv2d(self.filters * 4)
        self.down_4 = self.conv2d(self.filters * 8)
        self.down_5 = self.conv2d(self.filters * 8)
        self.down_6 = self.conv2d(self.filters * 8)
        self.down_7 = self.conv2d(self.filters * 8)

        # Upsampling
        self.up_1 = self.deconv2d(self.filters * 8, self.down_6)
        self.up_2 = self.deconv2d(self.filters * 8, self.down_5)
        self.up_3 = self.deconv2d(self.filters * 8, self.down_4)
        self.up_4 = self.deconv2d(self.filters * 4, self.down_3)
        self.up_5 = self.deconv2d(self.filters * 2, self.down_2)
        self.up_6 = self.deconv2d(self.filters, self.down_1)
        self.up_7 = Upsample(size=(2, 1))

        self.last = Conv2d(in_channels=2048, out_channels=2, kernel_size=4, padding=1)
        self.last_activation = Sigmoid()

    def conv2d(self, filters: int, kernel_size=(4, 1), batch_norm=True):
        model = Sequential()

        model.add_module("Conv2d", Conv2d(in_channels=2048, out_channels=filters,
                                          kernel_size=kernel_size, stride=(2, 1), padding=1))
        model.add_module("LeakyReLU", LeakyReLU(0.2))

        if batch_norm:
            model.add_module("BatchNorm2d", BatchNorm2d(num_features=1, momentum=0.8))

        return model

    def deconv2d(self, filters, skipped_layer, kernel_size=(4, 1), dropout_rate=False):
        model = Sequential()

        model.add_module("UpSample", Upsample(size=(2, 1)))
        model.add_module("Conv2d",
                         Conv2d(in_channels=2048, out_channels=filters, kernel_size=kernel_size, stride=(2, 1)))
        model.add_module("ReLU", ReLU())

        if dropout_rate:
            model.add_module("Dropout", Dropout(1.0))

        model.add_module("BatchNorm2d", BatchNorm2d(num_features=1, momentum=0.8))
        model.add_module("Concatenate", cat(model.get, skipped_layer))
        return model

    def forward(self, layer_input):
        layer_input = self.down_1(layer_input)
        layer_input = self.down_2(layer_input)
        layer_input = self.down_3(layer_input)
        layer_input = self.down_4(layer_input)
        layer_input = self.down_5(layer_input)
        layer_input = self.down_6(layer_input)
        layer_input = self.down_7(layer_input)

        layer_input = self.up_1(layer_input)
        layer_input = self.up_2(layer_input)
        layer_input = self.up_3(layer_input)
        layer_input = self.up_4(layer_input)
        layer_input = self.up_5(layer_input)
        layer_input = self.up_6(layer_input)
        layer_input = self.up_7(layer_input)

        layer_input = self.last(layer_input)
        layer_input = self.last_activation(layer_input)

        return layer_input
