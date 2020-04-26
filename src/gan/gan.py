from datetime import datetime

import numpy as np
import torch
from torch import Tensor, tensor
from torch.autograd import Variable
from torch.nn import MSELoss, L1Loss
from torch.optim import Adam

from src.gan.discriminator import Discriminator
from src.gan.generator import Generator
from src.utils.heatmaps import HeatMap
from src.utils.loader import Loader


class GAN:

    def __init__(self):
        self.percentage = 100  # percentual distribution of periodic samples
        self.file_rows = 2048
        self.file_cols = 1
        self.channels = 2  # 3 if bearing, 2 if just lat/lon
        self.file_shape = (self.file_rows, self.file_cols, self.channels)
        self.folder_to_save = 'Users/dhresko/Documents/Trajectories/generated'

        self.heat_map = HeatMap()
        self.data_loader = Loader(window=self.file_rows, portion=1000, days=3)

        self.distance_history = []

        # Building discriminator
        self.d_patch = (int(self.file_rows / 2 ** 4), 1, 1)

        self.discriminator = Discriminator(self.file_shape)
        self.optimizer_d = Adam(params=self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        # Building generator
        self.generator = Generator(self.file_shape)
        self.optimizer_g = Adam(params=self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        self.loss_mse = MSELoss()
        self.loss_l1 = L1Loss()

    def prepare_sequences(self, batch_size=1):
        network_input = []
        network_output = []

        for inp, out in self.data_loader.load_batch(batch_size):
            network_input.append(inp[0])
            network_output.append(out[0])

        return tensor(network_input, dtype=torch.double), tensor(network_output, dtype=torch.double)

    def sample_images(self, epoch, batch_i):
        (real_A, real_B) = self.prepare_sequences(1)
        fake_A = self.generator(real_A)

        real_A = real_A.reshape(self.file_rows, self.channels)
        fake_A = fake_A.reshape(self.file_rows, self.channels)
        real_B = real_B.reshape(self.file_rows, self.channels)

        avg = self.data_loader.save_generated_data(epoch, batch_i, real_B, real_A, fake_A, folder=self.folder_to_save,
                                                   save=0)
        self.heat_map.plot(real_B, savefolder=self.folder_to_save, epoch=epoch, save=0)
        self.heat_map.plot(real_A, savefolder=self.folder_to_save, epoch=epoch, save=0)
        self.heat_map.plot(fake_A, savefolder=self.folder_to_save, epoch=epoch)
        self.distance_history.append(({"Average distance": avg / 3}))

    def train(self, epochs: int, batch_size=1, sample_interval=50):

        # Adversarial ground truths
        valid = tensor(np.ones((batch_size,) + self.d_patch), requires_grad=False)
        fake = tensor(np.zeros((batch_size,) + self.d_patch), requires_grad=False)

        for epoch in range(epochs):
            real_A, real_B = self.prepare_sequences(batch_size)

            #  Train Generator
            self.optimizer_g.zero_grad()

            fake_A = self.generator(real_B)
            pred_fake = self.discriminator(fake_A, real_A)

            loss_mse = self.loss_mse(pred_fake, valid)
            loss_l1 = self.loss_l1(fake_A, tensor(real_B))

            # Total loss (100 is weight of L1 loss)
            loss_G = loss_mse + (100 * loss_l1)

            loss_G.backward()
            self.optimizer_g.step()

            #  Train Discriminator

            self.optimizer_d.zero_grad()

            # Real loss
            pred_real = self.discriminator(real_B, real_A)
            loss_real = self.loss_mse(pred_real, valid)

            # Fake loss
            pred_fake = self.discriminator(fake_A.detach(), real_B)
            loss_fake = self.loss_mse(pred_fake, fake)

            # Total loss
            loss_D = 0.5 * (loss_real + loss_fake)

            loss_D.backward()
            self.optimizer_d.step()
