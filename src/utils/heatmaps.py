import matplotlib.pyplot as plt
import numpy as np
import yaml


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

    def create_map(self, data_list: list, epoch: int, save=True, save_location=""):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        axes = (ax1, ax2, ax3)

        for index, data in enumerate(data_list):
            hmap = np.zeros((100, 100))

            data[:, 0] = data[:, 0] / 30 + 48.7
            data[:, 1] = data[:, 1] / 30 + 21.22
            # calculate "percentage" - the partition on 100x100 grid
            lat_offset = (data[:, 0] - self.min_lat) / self.lat_dist
            lon_offset = (data[:, 1] - self.min_lon) / self.lon_dist

            lat_offset *= 100
            lat_offset = lat_offset - 1
            lon_offset *= 100
            lon_offset = lon_offset - 1
            data[:, 0] = lat_offset
            data[:, 1] = lon_offset

            for tile in data:
                x = int(tile[0])
                y = int(tile[1])

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

            axes[index].imshow(hmap, cmap='hot', interpolation='nearest')
            axes[index].set_axis_off()

            if index == 0:
                axes[index].set_title('Real')
            elif index == 1:
                axes[index].set_title('Corruption')
            elif index == 2:
                axes[index].set_title('Generated')

        if save_location == "":
            if save:
                plt.savefig(f"{self.save_folder}/heatmap_{epoch}")
            plt.show()
        else:
            plt.savefig(f"{save_location}/heatmap_{epoch}")
