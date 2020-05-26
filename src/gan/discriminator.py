from torch import nn, cat
from torch.nn import Conv2d, Sequential, LeakyReLU, ZeroPad2d, BatchNorm2d

"""
PatchGan implementation
"""


class Discriminator(nn.Module):
    def __init__(self, file_shape: tuple, output_filters=8):
        super(Discriminator, self).__init__()

        self.model = Sequential(
            *self.build_block(file_shape[0] * 2, output_filters, normalization=False),
            *self.build_block(output_filters, output_filters * 2),
            *self.build_block(output_filters * 2, output_filters * 4),
            *self.build_block(output_filters * 4, output_filters * 8),

            ZeroPad2d((0, 0, 1, 0)),
            Conv2d(output_filters * 8, 1, kernel_size=(4, 1), padding=(1, 0), stride=1),
        )

    def build_block(self, in_filters: int, out_filters: int, normalization=True):
        layers = [Conv2d(in_filters, out_filters, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0)),
                  LeakyReLU(0.2, inplace=True)]

        if normalization:
            layers.append(BatchNorm2d(num_features=out_filters, momentum=0.8))

        return layers

    def forward(self, input_a, input_b):
        img_input = cat((input_a, input_b), 1)

        return self.model(img_input).double()
