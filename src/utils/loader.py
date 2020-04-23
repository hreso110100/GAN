import os
from math import sin, cos, sqrt, atan2, radians

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Loader:

    def __init__(self, channels=2, window=256, folder="/Users/dhresko/Documents/Trajectories/movementLogs", portion=0,
                 days=58):
        self.dir_data = folder
        self.portion = portion
        self.days = days
        self.fileList = self.getList()
        self.size = len(self.fileList)
        self.window = window
        self.channels = channels
        self.max_lat = 48.7171252
        self.min_lat = 48.7027684
        self.max_lon = 21.2497423
        self.min_lon = 21.228062
        self.lat_dist = (self.max_lat - self.min_lat)  # 0.015
        self.lon_dist = (self.max_lon - self.min_lon)  # 0.021

    def getList(self):
        names = np.sort(os.listdir(self.dir_data))  # periodic movement

        if self.portion > 0:
            names = names[:58 * self.portion]
            if self.days != 58:
                namelist = []
                for i in range(self.portion):
                    for suff in range(self.days):
                        namelist.append(names[58 * i + suff])
                return np.asarray(namelist)
        return names

    def getLen(self):
        return self.size

    def load_batch(self, batch_size, percentage=100):
        # percentage = percentual ammount of periodic trajectories
        """
        load trajectories from given folder
        window represents length of the loaded data, how many logs should be contained throughout one sample
        """
        #        if batch_size == 1:
        #            batch_size = 100
        if batch_size == 1:
            percentage = 100
        indexes = np.random.choice(self.size, int(batch_size * (percentage / 100)), replace=False)

        shape = (self.window, 1, self.channels)  # 3 if bearing is used

        for index in indexes:
            """Latitude interval:  [48.7027684, 48.7171252]
            Longitude interval:  [21.228062, 21.2497423]"""

            batch = []
            corr = []
            try:
                matrix = pd.read_csv(self.dir_data + "/" + self.fileList[index], header=None)
            except:
                print("Failed to load file " + self.fileList[index])
                continue
            matrix = matrix.drop(matrix.columns[0], axis=1)
            matrix = matrix.values
            matrix = matrix[:self.window, :]

            if self.channels == 3:
                distances = self.get_distance(matrix)

            # scale the values according to the lat/lon intervals
            lat = (matrix[:, 0] - 48.7) * 30
            matrix[:, 0] = lat
            lon = (matrix[:, 1] - 21.22) * 30
            matrix[:, 1] = lon
            matrix = np.nan_to_num(matrix)

            # additional channel
            if self.channels == 3:
                matrix = np.concatenate((matrix, distances), 1)

            matrix = matrix.reshape(shape)
            batch.append(matrix)
            corr = self.getCorruptFiles(batch)
            yield np.array(batch), np.array(corr)

    def load_generated(self, batch_size, cols, folder="./movementLogs", deli="\t"):
        """
        load trajectories from given folder
        window represents length of the loaded data, how many logs should be contained throughout one sample
        """
        #        if batch_size == 1:
        #            batch_size = 100
        self.dir_data = folder
        self.fileList = self.getList()
        self.size = len(self.fileList)
        # indexes = np.random.randint(low=0,high=self.size,size=batch_size)

        # shape = (self.window, 1, self.channels) #3 if bearing is used
        shape = (self.window, cols, self.channels)

        for index in range(batch_size):
            """Latitude interval:  [48.7027684, 48.7171252]
            Longitude interval:  [21.228062, 21.2497423]"""

            batch = []
            try:
                matrix = pd.read_csv(self.dir_data + "/" + self.fileList[index], header=None, delimiter=deli)
            except:
                print("Failed to load file " + self.fileList[index])
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

    def getCorruptFiles(self, files):
        """
        introduce random corruptions within files in dataset
        """
        corrupt = []
        size = len(files)
        i = 0
        for index in range(size):
            file = np.copy(files[index])
            indexes = np.random.choice(self.window, int(self.window * 0.95), replace=False)
            for i in range(int(self.window * 0.95)):  # remove random xy% points
                index = indexes[i]
                file[index, 0, 0] = 0
                file[index, 0, 1] = 0
            corrupt.append(file)
            i += 1
        return corrupt

    def plot_losses(self, history):
        hist = pd.DataFrame(history)
        plt.figure(figsize=(10, 5))
        for colnm in hist.columns:
            plt.plot(hist[colnm], label=colnm)
        plt.legend()
        plt.ylabel("loss")
        plt.xlabel("epochs")
        plt.show()

    def countDistance(self, lat1, lon1, lat2, lon2):

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
        # print( distance)
        return distance

    def save_generated_data(self, epoch, batch, files_B, files_A, fake_A, folder='Trajectories/release/pix2pix/',
                            save=1):

        #        files_A = files_A.reshape(3,self.window,2)
        #        fake_A = fake_A.reshape(3,self.window,2)
        #        files_B = files_B.reshape(3,self.window,2)
        avg = 0
        # larges = 0
        # print(files_B)
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
                avg += self.countDistance(df3[0][i], df3[1][i], df2[0][i], df2[1][i])

        # return larges/10,avg/10
        avg = avg / (self.window)
        if avg > 300:
            avg = 300
        return avg

    def save(self, batch, file, folder='Trajectories/release/pix2pix/'):

        df = pd.DataFrame()
        # print(file[:,0])
        df[0] = file[:, 0] / 30 + 48.7
        df[1] = file[:, 1] / 30 + 21.22
        df.to_csv(folder + str(batch) + '.csv')

    def get_distance(self, file):

        # if passing a generated file
        if file[0, 0] < 1:
            file[:, 0] = file[:, 0] / 30 + 48.7
            file[:, 1] = file[:, 1] / 30 + 21.22

        length = [[0]]
        for i in range(1, len(file)):
            dist = self.countDistance(file[i - 1, 0], file[i - 1, 1], file[i, 0], file[i, 1])
            length.append([dist])

        arr = np.where(np.array(length) > 300, 300, np.array(length))
        return arr / 300
