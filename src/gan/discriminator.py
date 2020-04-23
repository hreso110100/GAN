from torch import nn
from torch.nn import Conv2d, Sequential, LeakyReLU, BatchNorm2d


class Discriminator(nn.Module):

    def __init__(self, file_shape: tuple, filters: int):
        super(Discriminator, self).__init__()
        self.file_shape = file_shape
        self.filters = filters

        self.layer_1 = self.build_layer(filters, batch_normalization=False)
        self.layer_2 = self.build_layer(filters * 2)
        self.layer_3 = self.build_layer(filters * 4)
        self.layer_4 = self.build_layer(filters * 8)

        self.last_layer = Conv2d(in_channels=self.file_shape[0], out_channels=1, kernel_size=(4, 1))

    def build_layer(self, output_filters: int, kernel_size: tuple = (4, 1), batch_normalization=True):
        """
           This method builds individual Sequential model, based on parameters.
           output - Number of output filters.
           kernel - Kernel size.
           batch_normalization - True if you want to add.
        """
        model = Sequential()

        model.add_module("Conv2d", Conv2d(in_channels=self.file_shape[0], out_channels=output_filters,
                                          kernel_size=kernel_size, stride=(2, 1)))
        model.add_module("LeakyReLU", LeakyReLU(0.2))

        if batch_normalization:
            model.add_module("BatchNorm2d", BatchNorm2d(num_features=1, momentum=0.8))
        return model

    def forward(self, layer_input):
        layer_input = self.layer_1(layer_input)
        layer_input = self.layer_2(layer_input)
        layer_input = self.layer_3(layer_input)
        layer_input = self.layer_4(layer_input)
        layer_input = self.last_layer(layer_input)

        return layer_input
