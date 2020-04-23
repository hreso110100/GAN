import matplotlib.pyplot as plt
import numpy as np

from src.utils.loader import Loader


class HeatMap:

    def __init__(self):
        self.loader = Loader(window=2048)

    def get_files(self, batch_s, mode=0, folder="/Users/dhresko/Documents/Trajectories/movementLogs", perc=100, deli="\t"):
        """ Get 10 files per iteration """
        files = []
        if mode == 0:
            for batch_i, (data_batch) in enumerate(self.loader.load_batch(batch_s, percentage=perc)):
                files.append(data_batch)
        else:
            for batch_i, (data_batch) in enumerate(self.loader.load_generated(batch_s, 1, folder, deli)):
                files.append(data_batch)

        return files

    def prepare_sequences(self, files):
        """ Prepare the sequences used by the Neural Network """
        sequence_length = 2048
        network_input = []

        # create input sequences and the corresponding outputs
        f = 0
        for file in files:
            file = file[0][0]
            for i in range(1):
                # print(file.shape)
                sequence_in = file[0:sequence_length]
                network_input.append(sequence_in)
                if f == 0:
                    network_input[0] = sequence_in
                break

        n_patterns = len(network_input)

        # Reshape the input into a format compatible with LSTM layers
        network_input = np.reshape(network_input, (n_patterns, sequence_length, 2))

        return (network_input)

    def plotHeatMap(self, mode, folder):
        files = self.get_files(1000, mode, folder)
        prepared = self.prepare_sequences(files)

        hmap = np.zeros((100, 100))
        # print(prepared)
        for file in prepared:
            file[:, 0] = file[:, 0] / 30 + 48.7
            file[:, 1] = file[:, 1] / 30 + 21.22
            # print(file)
            # calculate "percentage" - the partition on 100x100 grid
            lat_offset = (file[:, 0] - self.loader.min_lat) / self.loader.lat_dist
            # print(matrix[:5,1])
            lon_offset = (file[:, 1] - self.loader.min_lon) / self.loader.lon_dist
            # print(lon_offset[:5])

            # so we can change type during offset counting
            lat_offset *= 100
            lat_offset = lat_offset.astype(int) - 1
            lon_offset *= 100
            lon_offset = lon_offset.astype(int) - 1
            file[:, 0] = lat_offset
            file[:, 1] = lon_offset
            for tile in file:
                # print(tile[0].astype(int))
                hmap[tile[0].astype(int), tile[1].astype(int)] += 1

        plt.imshow(hmap, cmap='hot', interpolation='nearest')
        plt.axis('off')
        plt.show()

    def plot(self, file, savefolder, epoch, save=1):
        hmap = np.zeros((100, 100))

        file[:, 0] = file[:, 0] / 30 + 48.7
        file[:, 1] = file[:, 1] / 30 + 21.22
        # calculate "percentage" - the partition on 100x100 grid
        lat_offset = (file[:, 0] - self.loader.min_lat) / self.loader.lat_dist
        # print(matrix[:5,1])
        lon_offset = (file[:, 1] - self.loader.min_lon) / self.loader.lon_dist
        # print(lon_offset[:5])

        # so we can change type during offset counting
        lat_offset *= 100
        lat_offset = lat_offset.astype(int) - 1
        lon_offset *= 100
        lon_offset = lon_offset.astype(int) - 1
        file[:, 0] = lat_offset
        file[:, 1] = lon_offset
        for tile in file:
            # print(tile[0].astype(int))
            x = tile[0].astype(int)
            y = tile[1].astype(int)
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
            plt.savefig(savefolder + "/" + str(epoch))
        plt.show()

    def plotHeatMaps(self, folder, mode=1, percentage=100, deli="\t"):
        files = self.get_files(5000, mode, folder, percentage, deli)
        prepared = self.prepare_sequences(files)

        hmap_to_remem = np.zeros((100, 100))

        hmap_to_remem_0 = np.zeros((100, 100))
        hmap_to_remem_4 = np.zeros((100, 100))
        hmap_to_remem_8 = np.zeros((100, 100))
        hmap_to_remem_12 = np.zeros((100, 100))
        hmap_to_remem_16 = np.zeros((100, 100))
        hmap_to_remem_20 = np.zeros((100, 100))
        hmap_to_remem_24 = np.zeros((100, 100))

        for file in prepared:
            hmap = np.zeros((100, 100))
            file[:, 0] = file[:, 0] / 30 + 48.7
            file[:, 1] = file[:, 1] / 30 + 21.22
            # calculate "percentage" - the partition on 100x100 grid
            lat_offset = (file[:, 0] - self.loader.min_lat) / self.loader.lat_dist
            lat_offset[:] = [x if x < 1 else 0.99 for x in lat_offset]
            lon_offset = (file[:, 1] - self.loader.min_lon) / self.loader.lon_dist
            lon_offset[:] = [x if x < 1 else 0.99 for x in lon_offset]

            # so we can change type during offset counting
            lat_offset *= 100
            lat_offset = lat_offset.astype(int) - 1
            lon_offset *= 100
            lon_offset = lon_offset.astype(int) - 1
            file[:, 0] = lat_offset
            file[:, 1] = lon_offset
            i = 0
            for tile in file:
                # print(tile[0].astype(int))
                hmap[tile[0].astype(int), tile[1].astype(int)] += 1
                hmap_to_remem[tile[0].astype(int), tile[1].astype(int)] += 1
                # remember time series:
                if i == 5:
                    hmap_to_remem_0[tile[0].astype(int), tile[1].astype(int)] += 1
                if i == 341:
                    hmap_to_remem_4[tile[0].astype(int), tile[1].astype(int)] += 1
                if i == 682:
                    hmap_to_remem_8[tile[0].astype(int), tile[1].astype(int)] += 1
                if i == 1024:
                    hmap_to_remem_12[tile[0].astype(int), tile[1].astype(int)] += 1
                if i == 1365:
                    hmap_to_remem_16[tile[0].astype(int), tile[1].astype(int)] += 1
                if i == 1706:
                    hmap_to_remem_20[tile[0].astype(int), tile[1].astype(int)] += 1
                if i == 2040:
                    hmap_to_remem_24[tile[0].astype(int), tile[1].astype(int)] += 1
                i += 1

        for row in hmap_to_remem:
            row[:] = [x if x < 4000 else 4000 for x in row]  # replacing high density to better read the map
        self.plot_map(hmap_to_remem)

    def plot_map(self, hmap):
        plt.imshow(hmap, cmap='hot', interpolation='nearest')
        plt.axis('off')
        plt.show()
