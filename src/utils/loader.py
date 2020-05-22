import os

import numpy as np
import pandas as pd
import yaml
from torch import tensor


class Loader:

    def __init__(self, channels=2, window=256, portion=0, days=58):
        with open(f"../src/config/model_config.yml", 'r') as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)

        self.dataset_folder = self.config["folders"]["dataset"]
        self.generated_folder = self.config["folders"]["generated"]
        self.portion = portion
        self.days = days
        self.fileList = self.get_files()
        self.number_of_files = len(self.fileList)
        self.window = window
        self.channels = channels
        self.shape = (self.channels, self.window, 1)

    def get_files(self) -> np.ndarray:
        """
        Loading files in dataset dir.

        :return: List of files in dataset dir.
        """

        files = np.sort(os.listdir(self.dataset_folder))

        if self.portion > 0:
            files = files[:58 * self.portion]

            if self.days != 58:
                namelist = []
                for i in range(self.portion):
                    for suff in range(self.days):
                        namelist.append(files[58 * i + suff])

                return np.asarray(namelist)

        return files

    def load_batch(self, batch_size=1, percentage=100):
        """
        Load trajectories from given folder.
        Window represents length of the loaded data, how many logs should be contained throughout one sample.

        :param batch_size: Size of batch.
        :param percentage: Percentual amount of periodic trajectories.

        :return: Tuple containing original data and corrupted data.
        """

        if batch_size == 1:
            percentage = 100

        # Selecting random data files
        chosen_files = np.random.choice(self.number_of_files, int(batch_size * (percentage / 100)), replace=False)

        for _, chosen_index in enumerate(chosen_files):
            batch = []

            try:
                loaded_data = pd.read_csv(self.dataset_folder + "/" + self.fileList[chosen_index], header=None)
            except FileNotFoundError:
                print(f"LOGGER: Cannot load given file {self.fileList[chosen_index]}.")
                continue

            loaded_data = self.drop_timestamp(loaded_data)
            loaded_data = self.scale_data(loaded_data)

            loaded_data = loaded_data.reshape(self.shape)
            batch.append(loaded_data)

            noise = self.add_corruption(batch)

            yield np.array(batch), np.array(noise)

    def scale_data(self, data) -> np.array:
        """
        Rescaling data based on latitude and longitude range.

        :param data: Data to scale.

        :return: Scaled data.
        """

        lat = (data[:, 0] - 48.7) * 30
        data[:, 0] = lat
        lon = (data[:, 1] - 21.22) * 30
        data[:, 1] = lon
        data = np.nan_to_num(data)

        return data

    def drop_timestamp(self, data) -> np.array:
        """
        Dropping timestamp from dataset.

        :param data: Data to transform.
        :return: Transformed data.
        """

        data = data.drop(data.columns[0], axis=1)
        data = data.values
        data = data[:self.window, :]

        return data

    def add_corruption(self, files: list) -> list:
        """
        Adding random corruption within files in dataset.

        :param files: List of dataset files.

        :return: Corrupted list of files.
        """

        corrupt = []
        remove_ratio = int(self.window * 0.95)
        # Choosing N numbers of random rows to be deleted, based on remove_ratio

        for file in files:
            file = file.reshape(2048, 1, 2)
            file_copy = np.copy(file)
            remove_indexes = np.random.choice(self.window, remove_ratio, replace=False)

            for index in range(remove_ratio):  # remove random rows in given file
                file_copy[remove_indexes[index], 0, 0] = 0
                file_copy[remove_indexes[index], 0, 1] = 0
            corrupt.append(file_copy.reshape(2, 2048, 1))

        return corrupt

    def load_generated(self, batch_size, cols, folder="./movementLogs", deli="\t"):
        """
        load trajectories from given folder
        window represents length of the loaded data, how many logs should be contained throughout one sample
        """

        self.dataset_folder = folder
        self.fileList = self.get_files()
        self.number_of_files = len(self.fileList)

        shape = (self.window, cols, self.channels)

        for index in range(batch_size):
            batch = []
            try:
                matrix = pd.read_csv(self.dataset_folder + "/" + self.fileList[index], header=None, delimiter=deli)
            except FileNotFoundError:
                print("Cannot load file " + self.fileList[index])
                continue
            matrix = matrix.drop(matrix.columns[0], axis=0)
            #            matrix = matrix.drop(matrix.columns[3], axis=1)
            null = self.drop_timestamp(matrix)
            # scale the values according to the lat/lon intervals

            null = self.scale_data(matrix)

            matrix2 = matrix.reshape(shape)
            batch.append(matrix2)
            # print(np.array(batch))
            yield np.array(batch), matrix

    def save_data(self, epoch: int, batch: int, corrupted: tensor, real: tensor, fake: tensor, save=True):
        """
        Saving data.

        :param epoch: Number of epochs.
        :param batch: Number of batches.
        :param corrupted: Corrupted data.
        :param real: Real data.
        :param fake: Generated data.
        :param save: Boolean if save or not.
        """

        self.save(epoch, batch, corrupted, self.generated_folder, "corrupted", save)
        self.save(epoch, batch, real, self.generated_folder, "real", save)
        self.save(epoch, batch, fake, self.generated_folder, "generated", save)

    def save(self, epoch: int, batch: int, data: tensor, folder: str, file_name: str, save=True):
        """
        Base method for saving.

        :param epoch: Number of epochs.
        :param batch: Number of batches.
        :param data: Data to save.
        :param folder: Folder where data will be saved.
        :param file_name: Name of the final file. File will be CSV by default.
        :param save: Boolean if save or not.
        """

        df = pd.DataFrame()
        data = data.detach()

        df[0] = data[:, 0] / 30 + 48.7
        df[1] = data[:, 1] / 30 + 21.22

        if save:
            df.to_csv(f"{folder}/{str(epoch)}_{str(batch)}_{file_name}.csv")
