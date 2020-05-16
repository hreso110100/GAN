import matplotlib.pyplot as plt
import numpy as np
import yaml
from torch import tensor


class HeatMap:

    def __init__(self):
        with open(f"../src/config/model_config.yml", 'r') as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)

        self.save_folder = self.config["folders"]["heatmaps"]
        self.max_lat = 48.7171252
        self.min_lat = 48.7027684
        self.max_lon = 21.2497423
        self.min_lon = 21.228062
        self.lat_dist = (self.max_lat - self.min_lat)  # 0.015
        self.lon_dist = (self.max_lon - self.min_lon)  # 0.021

    def create_map(self, data: tensor, data_type: str, epoch: int, save=True):
        hmap = np.zeros((100, 100))

        data[:, 0] = data[:, 0] / 30 + 48.7
        data[:, 1] = data[:, 1] / 30 + 21.22
        # calculate "percentage" - the partition on 100x100 grid
        lat_offset = (data[:, 0] - self.min_lat) / self.lat_dist
        lon_offset = (data[:, 1] - self.min_lon) / self.lon_dist

        # so we can change type during offset counting
        lat_offset *= 100
        lat_offset = lat_offset.int() - 1
        lon_offset *= 100
        lon_offset = lon_offset.int() - 1
        data[:, 0] = lat_offset
        data[:, 1] = lon_offset

        for tile in data:
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
            plt.savefig(f"{self.save_folder}/heatmap_{data_type}_{epoch}")
        plt.show()
