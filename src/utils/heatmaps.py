import matplotlib.pyplot as plt
import numpy as np

from src.utils.loader import Loader


class HeatMap:

    def __init__(self):
        self.loader = Loader(window=2048)
        self.save_folder = "D://dHresko/heatmaps"
        self.max_lat = 48.7171252
        self.min_lat = 48.7027684
        self.max_lon = 21.2497423
        self.min_lon = 21.228062
        self.lat_dist = (self.max_lat - self.min_lat)  # 0.015
        self.lon_dist = (self.max_lon - self.min_lon)  # 0.021

    def plot(self, file, epoch, save=1):
        hmap = np.zeros((100, 100))

        file[:, 0] = file[:, 0] / 30 + 48.7
        file[:, 1] = file[:, 1] / 30 + 21.22
        # calculate "percentage" - the partition on 100x100 grid
        lat_offset = (file[:, 0] - self.min_lat) / self.lat_dist
        # print(matrix[:5,1])
        lon_offset = (file[:, 1] - self.min_lon) / self.lon_dist
        # print(lon_offset[:5])

        # so we can change type during offset counting
        lat_offset *= 100
        lat_offset = lat_offset - 1
        lon_offset *= 100
        lon_offset = lon_offset - 1
        file[:, 0] = lat_offset
        file[:, 1] = lon_offset
        for tile in file:
            x = tile[0].int()
            y = tile[1].int()
            if x > 99:
                x = 99
            if y > 99:
                y = 99
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            hmap[x, y] += 1
        for row in hmap:
            row[:] = [x if x < 50 else 50 for x in row]  # replacing high density to better read the map

        plt.imshow(hmap, cmap='hot', interpolation='nearest')
        plt.axis('off')
        if save:
            plt.savefig(self.save_folder + "/" + str(epoch))
        plt.show()
