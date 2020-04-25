from datetime import datetime

import numpy as np
from torch import tensor
from torch.nn import MSELoss, Module, L1Loss
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

        tensor_A = tensor(self.file_shape)
        tensor_B = tensor(self.file_shape)

        # Building discriminator
        patch = int(self.file_rows / 2 ** 4)
        self.disc_patch = (patch, 1, 1)

        self.discriminator = Discriminator(self.file_shape, filters=8)
        self.discriminator.train(mode=False)
        self.optimizer_d = Adam(params=self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.loss_d = MSELoss()

        # Building generator
        self.generator = Generator(self.file_shape, filters=8)

        # Building combined model
        fake = self.generator(tensor_B)
        valid = self.discriminator([fake, tensor_B])

        self.combined = Module(inputs=[tensor_A, tensor_B], outputs=[valid, fake])
        self.optimizer_combined = Adam(params=self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.loss_mse_combined = MSELoss()
        self.loss_mae_combined = L1Loss()

    def prepare_sequences(self, batch_size=1):
        network_input = []
        network_output = []

        for batch_i, (in_, out) in enumerate(self.data_loader.load_batch(batch_size)):
            network_input.append(in_[0])
            network_output.append(out[0])

        return np.array(network_input), np.array(network_output)

    def sample_images(self, epoch, batch_i):
        (files_A, files_B) = self.prepare_sequences(1)
        fake_A = self.generator.predict(files_B)

        files_A = files_A.reshape(self.file_rows, self.channels)
        fake_A = fake_A.reshape(self.file_rows, self.channels)
        files_B = files_B.reshape(self.file_rows, self.channels)
        avg = self.data_loader.save_generated_data(epoch, batch_i, files_B, files_A, fake_A, folder=self.folder_to_save,
                                                   save=0)
        self.heat_map.plot(files_B, savefolder=self.folder_to_save, epoch=epoch, save=0)
        self.heat_map.plot(files_A, savefolder=self.folder_to_save, epoch=epoch, save=0)
        self.heat_map.plot(fake_A, savefolder=self.folder_to_save, epoch=epoch)
        self.distance_history.append(({"Average distance": avg / 3}))

    def train(self, epochs: int, batch_size=1, sample_interval=50):
        start_time = datetime.now()

        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            pass
