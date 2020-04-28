import datetime

import numpy as np
from torch import tensor
from torch.nn import MSELoss, L1Loss
from torch.optim import Adam

from src.gan.discriminator import Discriminator
from src.gan.generator import Generator
from src.utils.distance import Distance
from src.utils.heatmaps import HeatMap
from src.utils.loader import Loader


class GAN:

    def __init__(self):
        self.percentage = 100
        self.file_rows = 2048
        self.file_cols = 1
        self.channels = 2  # 3 if bearing, 2 if just lat/lon
        self.file_shape = (self.channels, self.file_rows, self.file_cols)

        self.heat_map = HeatMap()
        self.distance = Distance(window=self.file_rows)
        self.data_loader = Loader(window=self.file_rows, portion=1000, days=3)

        self.distance_history = []
        self.losses = []

        # Building discriminator
        self.d_patch = (1, int(self.file_rows / 2 ** 4), 1)

        self.discriminator = Discriminator(self.file_shape)
        self.optimizer_d = Adam(params=self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        # Building generator
        self.generator = Generator(self.file_shape)
        self.optimizer_g = Adam(params=self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        # Building losses
        self.loss_mse = MSELoss()
        self.loss_l1 = L1Loss()

    def prepare_sequences(self, batch_size=1) -> tuple:
        """
        Preparing sequences of real and corrupted data.

        :param batch_size: Size of batch.

        :return: Tuple of real and corrupted data.
        """

        real_data = []
        corrupted_data = []

        for _, (real, corrupted) in enumerate(self.data_loader.load_batch(batch_size)):
            real_data.append(real[0])
            corrupted_data.append(corrupted[0])

        return tensor(real_data).float(), tensor(corrupted_data).float()

    def sample_images(self, epoch, batch_size):
        """
        Continuos saving of data.

        :param epoch: Current epoch.
        :param batch_size: Batch size.
        """

        real, corrupted = self.prepare_sequences()
        fake = self.generator(real)

        real = real.reshape(self.file_rows, self.channels)
        fake = fake.reshape(self.file_rows, self.channels)
        corrupted = corrupted.reshape(self.file_rows, self.channels)

        self.data_loader.save_data(epoch, batch_size, corrupted, real, fake)
        avg_distance = self.distance.get_avg_distance(fake, real)

        self.heat_map.plot(corrupted, epoch=epoch)
        self.heat_map.plot(real, epoch=epoch)
        self.heat_map.plot(fake, epoch=epoch)
        self.distance_history.append(({"Average distance": avg_distance / 3}))

    def train(self, epochs: int, batch_size=1, sample_interval=50):
        start_time = datetime.datetime.now()

        # Adversarial ground truths
        valid = tensor(np.ones((batch_size,) + self.d_patch), requires_grad=False)
        fake = tensor(np.zeros((batch_size,) + self.d_patch), requires_grad=False)

        for epoch in range(epochs):
            real_A, real_B = self.prepare_sequences(batch_size)

            #  Train Generator
            self.optimizer_g.zero_grad()

            fake_B = self.generator(real_A)
            pred_fake = self.discriminator(fake_B, real_A)

            loss_mse = self.loss_mse(pred_fake, valid)
            loss_l1 = self.loss_l1(fake_B, real_B)

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
            pred_fake = self.discriminator(fake_B.detach(), real_A)
            loss_fake = self.loss_mse(pred_fake, fake)

            # Total loss
            loss_D = 0.5 * (loss_real + loss_fake)

            loss_D.backward()
            self.optimizer_d.step()

            elapsed_time = datetime.datetime.now() - start_time
            print(f"[Epoch {epoch}/{epochs}] [D loss: {loss_D}] [G loss: {loss_G}] time: {elapsed_time}")

            if epoch % sample_interval == 0:
                self.sample_images(epoch, batch_size)
                self.losses.append({"D": loss_D[0], "G": loss_G[0]})

    def plot_loss(self):
        pass

    def save_models(self):
        pass

    # def plot_losses(self, history):
    #     hist = pd.DataFrame(history)
    #     plt.figure(figsize=(10, 5))
    #
    #     for colnm in hist.columns:
    #         plt.plot(hist[colnm], label=colnm)
    #
    #     plt.legend()
    #     plt.ylabel("loss")
    #     plt.xlabel("epochs")
    #     plt.show()
