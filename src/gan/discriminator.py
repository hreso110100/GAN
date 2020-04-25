from torch import nn, cat
from torch.nn import Conv2d, Sequential, LeakyReLU, BatchNorm2d


class Discriminator(nn.Module):
    def __init__(self, file_shape: tuple, out_filters=8):
        super(Discriminator, self).__init__()

        self.model = Sequential(
            *self.build_block(file_shape[0] * 2, out_filters, normalization=False),
            *self.build_block(out_filters, out_filters * 2),
            *self.build_block(out_filters * 2, out_filters * 4),
            *self.build_block(out_filters * 4, out_filters * 8),

            nn.Conv2d(out_filters * 8, 1, kernel_size=(4, 1), padding=1)
        )

    def build_block(self, in_filters: int, out_filters: int, normalization=True):
        layers = [Conv2d(in_filters, out_filters, kernel_size=(4, 1), stride=2, padding=1)]

        if normalization:
            layers.append(BatchNorm2d(num_features=out_filters, momentum=0.8))

        layers.append(LeakyReLU(0.2, inplace=True))

        return layers

    def forward(self, dataset_data, generated_data):
        img_input = cat((dataset_data, generated_data), 1)

        return self.model(img_input)
