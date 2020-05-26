import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import tensor
from torch.nn import MSELoss, L1Loss
from torch.optim import Adam

from src.gan.discriminator import Discriminator
from src.gan.generator import Generator
from src.utils.distance import Distance
from src.utils.heatmaps import HeatMap
from src.utils.loader import Loader


def weights_init(model):
    """
    Init weights for CNN layers.

    :param model: Model to be initialized
    """
    classname = model.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.xavier_uniform_(model.weight)
        if model.bias is not None:
            torch.nn.init.zeros_(model.bias)


class GAN:

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        with open(f"../src/config/model_config.yml", 'r') as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)

        self.samples_folder = self.config["folders"]["generated_samples"]

        self.percentage = 100
        self.file_rows = 2048
        self.file_cols = 1
        self.channels = 2
        self.file_shape = (self.channels, self.file_rows, self.file_cols)

        self.heat_map = HeatMap()
        self.distance = Distance(window=self.file_rows)
        self.data_loader = Loader(shape=self.file_shape, portion=1000, days=3)

        self.distance_history = []
        self.losses = []

        # Building losses
        self.loss_mse = MSELoss()
        self.loss_l1 = L1Loss()

        # Building discriminator
        self.d_patch = (1, int(self.file_rows // 2 ** 4), 1)

        self.discriminator = Discriminator(self.file_shape).to(self.device)
        self.discriminator.apply(weights_init)
        self.optimizer_d = Adam(params=self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        # Building generator
        self.generator = Generator(self.file_shape).to(self.device)
        self.generator.apply(weights_init)
        self.optimizer_g = Adam(params=self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def prepare_sequences(self, batch_size=1) -> tuple:
        """
        Preparing sequences of real and corrupted data.

        :param batch_size: Size of batch.

        :return: Tuple of real and corrupted data.
        """

        real_data = []
        corrupted_data = []

        for _, (real, corrupted) in enumerate(self.data_loader.load_batch(batch_size, self.percentage)):
            real_data.append(real[0])
            corrupted_data.append(corrupted[0])

        return tensor(real_data, device=self.device).float(), tensor(corrupted_data, device=self.device).float()

    def sample_images(self, epoch, batch_size):
        """
        Continuous saving of data.

        :param epoch: Current epoch.
        :param batch_size: Batch size.
        """

        real, corrupted = self.prepare_sequences()
        fake = self.generator(corrupted)

        fake = fake.detach().cpu().numpy().swapaxes(1, 2).reshape(self.file_rows, self.channels)
        real = real.detach().cpu().numpy().swapaxes(1, 2).reshape(self.file_rows, self.channels)
        corrupted = corrupted.detach().cpu().numpy().swapaxes(1, 2).reshape(self.file_rows, self.channels)

        self.data_loader.save_data(epoch, batch_size, corrupted, real, fake)
        self.heat_map.create_map(data_list=[real, corrupted, fake], epoch=epoch)

        avg_distance = self.distance.get_avg_distance(fake, real)
        self.distance_history.append(({"Average distance": avg_distance}))

    def train(self, epochs: int, batch_size: int, sample_interval: int):
        start_time = datetime.datetime.now()

        # Adversarial ground truths
        valid = tensor(np.ones((batch_size,) + self.d_patch), requires_grad=False, device=self.device)
        fake = tensor(np.zeros((batch_size,) + self.d_patch), requires_grad=False, device=self.device)

        for epoch in range(epochs):
            real_A, real_B = self.prepare_sequences(batch_size)
            fake_A = self.generator(real_B)

            #  Train Generator
            for param in self.discriminator.parameters():
                param.requires_grad_(False)

            self.optimizer_g.zero_grad()

            pred_fake = self.discriminator(fake_A, real_B)

            loss_mse = self.loss_mse(pred_fake, valid).double()
            loss_l1 = self.loss_l1(fake_A, real_A).double()

            # Total loss (100 is weight of L1 loss)
            loss_G = loss_mse + (100 * loss_l1)

            loss_G.backward()
            self.optimizer_g.step()

            #  Train Discriminator
            for param in self.discriminator.parameters():
                param.requires_grad_(True)

            self.optimizer_d.zero_grad()

            # Real loss
            pred_real = self.discriminator(real_A, real_B)
            loss_real = self.loss_mse(pred_real, valid)

            # Fake loss
            pred_fake = self.discriminator(fake_A.detach(), real_B)
            loss_fake = self.loss_mse(pred_fake, fake)

            # Total loss
            loss_D = 0.5 * (loss_real + loss_fake)

            loss_D.backward()
            self.optimizer_d.step()

            elapsed_time = datetime.datetime.now() - start_time
            self.losses.append({"D": loss_D, "G": loss_G})
            print(f"[Epoch {epoch}/{epochs}] [D loss: {loss_D}] [G loss: {loss_G}] time: {elapsed_time}")

            if epoch % sample_interval == 0:
                self.sample_images(epoch, batch_size)

        self.plot_loss(self.losses)
        self.generate_samples(100)
        self.save_models()

    def plot_loss(self, loss_list: list):
        """
        Plot losses of discriminator and generator.
        """

        plt.figure(figsize=(12, 5))
        loss_G = []
        loss_D = []

        for loss in loss_list:
            loss_G.append(loss["G"].detach().numpy())
            loss_D.append(loss["D"].detach().numpy())

        plt.plot(loss_G, label="Generator")
        plt.plot(loss_D, label="Discriminator")

        plt.title("Discriminator and generator loss")
        plt.ylabel("loss")
        plt.xlabel("epochs")
        plt.show()

    def generate_samples(self, samples: int):
        """
        Generate N number of fake trajectories and store them on disc.

        :param samples: Number of samples to generate
        """
        for i in range(samples):
            print(f"LOGGER: Generating sample {i+1}/{samples}.")
            real, corrupted = self.prepare_sequences(1)
            fake = self.generator(corrupted)

            fake = fake.detach().cpu().numpy().swapaxes(1, 2).reshape(self.file_rows, self.channels)
            real = real.detach().cpu().numpy().swapaxes(1, 2).reshape(self.file_rows, self.channels)
            corrupted = corrupted.detach().cpu().numpy().swapaxes(1, 2).reshape(self.file_rows, self.channels)

            self.heat_map.create_map(data_list=[real, corrupted, fake], epoch=i, save_location=self.samples_folder)

    def save_models(self):
        # TODO
        if not os.path.exists('../models'):
            os.makedirs('../models')
