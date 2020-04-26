import os
from math import sin, cos, sqrt, atan2, radians

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Loader:

    def __init__(self, channels=2, window=256, portion=0, days=58):
        self.dir_data = "/Users/dhresko/Documents/Trajectories/movementLogs"
        self.portion = portion
        self.days = days
        self.fileList = self.get_files()
        self.number_of_files = len(self.fileList)
        self.window = window
        self.channels = channels
        self.shape = (self.window, 1, self.channels)
        self.max_lat = 48.7171252
        self.min_lat = 48.7027684
        self.max_lon = 21.2497423
        self.min_lon = 21.228062
        self.lat_dist = (self.max_lat - self.min_lat)  # 0.015
        self.lon_dist = (self.max_lon - self.min_lon)  # 0.021

    def get_files(self) -> list:
        """
        Loading files in dataset dir.

        :return: List of files in dataset dir.
        """

        files = np.sort(os.listdir(self.dir_data))

        if self.portion > 0:
            files = files[:self.days * self.portion]

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

        for index, chosen_index in enumerate(chosen_files):
            print(f"LOGGER: Loading file {index + 1} / {len(chosen_files)}.")
            batch = []

            try:
                loaded_data = pd.read_csv(self.dir_data + "/" + self.fileList[chosen_index], header=None)
            except FileNotFoundError:
                print(f"LOGGER: Cannot load given file {self.fileList[chosen_index]}.")
                continue

            # Timestamp will be dropped,
            loaded_data = loaded_data.drop(loaded_data.columns[0], axis=1)
            loaded_data = loaded_data.values
            loaded_data = loaded_data[:self.window, :]

            # Scale the values according to the lat/lon intervals
            lat = (loaded_data[:, 0] - 48.7) * 30
            loaded_data[:, 0] = lat
            lon = (loaded_data[:, 1] - 21.22) * 30
            loaded_data[:, 1] = lon
            loaded_data = np.nan_to_num(loaded_data)

            loaded_data = loaded_data.reshape(self.shape)
            batch.append(loaded_data)

            noise = self.add_corruption(batch)

            yield np.array(batch), np.array(noise)

    def add_corruption(self, files: list) -> list:
        """
        Adding random corruption within files in dataset.

        :param files: List of dataset files.

        :return: Corrupted list of files.
        """

        corrupt = []
        remove_ratio = int(self.window * 0.95)
        # Choosing N numbers of random rows to be deleted, based on remove_ratio
        remove_indexes = np.random.choice(self.window, remove_ratio, replace=False)

        for file in files:
            file_copy = np.copy(file)

            for remove_index in remove_indexes:  # remove random rows in given file
                file_copy[remove_index, 0, 0] = 0
                file_copy[remove_index, 0, 1] = 0
            corrupt.append(file_copy)

        return corrupt

    def count_distance(self, lat1: float, lon1: float, lat2: float, lon2: float):
        """
        Calculating distance between two points.

        :param lat1: Latitude of point A.
        :param lon1: Longitude of point A.
        :param lat2: Latitude of point B.
        :param lon2: Longitude of point B.

        :return: Distance between two points.
        """

        R = 6373000  # earth radius in meters

        lat1 = radians(lat1)
        lon1 = radians(lon1)
        lat2 = radians(lat2)
        lon2 = radians(lon2)

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        distance = R * c

        return distance

    def plot_losses(self, history):
        hist = pd.DataFrame(history)
        plt.figure(figsize=(10, 5))

        for colnm in hist.columns:
            plt.plot(hist[colnm], label=colnm)

        plt.legend()
        plt.ylabel("loss")
        plt.xlabel("epochs")
        plt.show()

    def load_generated(self, batch_size, cols, folder="./movementLogs", deli="\t"):
        """
        load trajectories from given folder
        window represents length of the loaded data, how many logs should be contained throughout one sample
        """
        #        if batch_size == 1:
        #            batch_size = 100
        self.dir_data = folder
        self.fileList = self.get_files()
        self.number_of_files = len(self.fileList)

        shape = (self.window, cols, self.channels)

        for index in range(batch_size):
            """Latitude interval:  [48.7027684, 48.7171252]
            Longitude interval:  [21.228062, 21.2497423]"""

            batch = []
            try:
                matrix = pd.read_csv(self.dir_data + "/" + self.fileList[index], header=None, delimiter=deli)
            except FileNotFoundError:
                print("Cannot load file " + self.fileList[index])
                continue
            matrix = matrix.drop(matrix.columns[0], axis=0)
            #            matrix = matrix.drop(matrix.columns[3], axis=1)
            matrix = matrix.drop(matrix.columns[0], axis=1)
            matrix = matrix.values
            matrix = matrix[:self.window, :]
            # scale the values according to the lat/lon intervals

            lat = (matrix[:, 0] - 48.7) * 30
            matrix[:, 0] = lat
            lon = (matrix[:, 1] - 21.22) * 30
            matrix[:, 1] = lon
            matrix = np.nan_to_num(matrix)

            matrix2 = matrix.reshape(shape)
            batch.append(matrix2)
            # print(np.array(batch))
            yield np.array(batch), matrix

    def save_generated_data(self, epoch, batch, files_B, files_A, fake_A, folder='Trajectories/release/pix2pix/',
                            save=1):

        #        files_A = files_A.reshape(3,self.window,2)
        #        fake_A = fake_A.reshape(3,self.window,2)
        #        files_B = files_B.reshape(3,self.window,2)
        avg = 0

        for i in range(1):
            df = pd.DataFrame()
            df[0] = files_B[:, 0] / 30 + 48.7
            df[1] = files_B[:, 1] / 30 + 21.22
            if save:
                df.to_csv(folder + str(epoch) + '_' + str(batch) + '_base.csv', sep='\t')

            df2 = pd.DataFrame()
            df2[0] = fake_A[:, 0] / 30 + 48.7
            df2[1] = fake_A[:, 1] / 30 + 21.22
            if save:
                df2.to_csv(folder + str(epoch) + '_' + str(batch) + '_gener.csv', sep='\t')

            df3 = pd.DataFrame()
            df3[0] = files_A[:, 0] / 30 + 48.7
            df3[1] = files_A[:, 1] / 30 + 21.22
            if save:
                df3.to_csv(folder + str(epoch) + '_' + str(batch) + '_true.csv', sep='\t')

            for i in range(self.window):
                avg += self.count_distance(df3[0][i], df3[1][i], df2[0][i], df2[1][i])

        # return larges/10,avg/10
        avg = avg / self.window
        if avg > 300:
            avg = 300
        return avg

    def save(self, batch, file, folder='Trajectories/release/pix2pix/'):

        df = pd.DataFrame()
        df[0] = file[:, 0] / 30 + 48.7
        df[1] = file[:, 1] / 30 + 21.22
        df.to_csv(folder + str(batch) + '.csv')
