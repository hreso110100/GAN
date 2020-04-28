from math import sin, cos, sqrt, atan2, radians


class Distance:

    def __init__(self, window=256):
        self.window = window

    def get_avg_distance(self, generated, real):
        """
        Calculate distance between generated and real point and make average.

        :param generated: Generated data.
        :param real: Real data.

        :return: Average distance.
        """

        avg_distance = 0

        for i in range(self.window):
            avg_distance += self.count_distance(real[i][0], real[i][1], generated[i][0], generated[i][1])

        avg_distance = avg_distance / self.window

        if avg_distance > 300:
            avg_distance = 300

        return avg_distance

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
