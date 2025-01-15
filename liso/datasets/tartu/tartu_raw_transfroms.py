import csv
import numpy as np
from pyproj import Transformer


class TartuRawTransfroms:
    def __init__(self, csv_path):
        self.locations = {}
        self.transfroms = {}

        self.origin_lat = 58.385345
        self.origin_lon = 26.726272

        # Set up a transformer for WGS84 to UTM
        self.wgs2utm_transformer = Transformer.from_crs("epsg:4326", "epsg:32635")  # UTM Zone 35N
        self.origin_easting, self.origin_northing = self.wgs2utm_transformer.transform(self.origin_lat, self.origin_lon)

        # find out zone number, multiply with width of zone (6 degrees) and add 3 degrees to get central meridian
        self.central_meridian = (self.origin_lon // 6.0) * 6.0 + 3.0

        self.load_csv(csv_path)

        assert len(self.locations) > 0
        assert len(self.transfroms) > 0


    def load_csv(self, csv_path):
        with open(csv_path, mode='r', newline='') as file:
            reader = csv.reader(file, delimiter=',')
            next(reader, None) # skip header
            for row in reader:
                self.locations[int(row[0])] = row[1:]

                longitude, latitude, height, azimuth, roll, pitch = row[2:]

                map_T_bl = self.create_map_T_bl_matrix(float(longitude), float(latitude), float(height), float(azimuth), float(roll), float(pitch))
                self.transfroms[int(row[0])] = map_T_bl


    def create_map_T_bl_matrix(self, longitude, latitude, height, azimuth, roll, pitch):
        x, y = self.convert_WGS84_to_UTM(latitude, longitude)
        corrected_azimuth = self.correct_azimuth(latitude, longitude, azimuth)
        rotation_matrix = self.rpy_to_rotation_matrix(roll, pitch, corrected_azimuth)
        translation_matrix = np.array([x, y, height]).reshape(3, 1)

        map_T_bl = np.vstack((np.hstack((rotation_matrix, translation_matrix)), [0, 0, 0, 1]))

        return map_T_bl


    def convert_WGS84_to_UTM(self, latitude, longitude):
        easting, northing = self.wgs2utm_transformer.transform(latitude, longitude)

        return easting - self.origin_easting, northing - self.origin_northing


    def correct_azimuth(self, lat, lon, azimuth):
        # calculate grid convergence and use to correct the azimuth
        # https://gis.stackexchange.com/questions/115531/calculating-grid-convergence-true-north-to-grid-north
        a = np.tan(np.radians(lon - self.central_meridian))
        b = np.sin(np.radians(lat))
        correction = np.degrees(np.arctan(a * b))

        # TODO compare with CA = (Lambda - LambdaCM) * sin Theta

        return azimuth - correction


    def rpy_to_rotation_matrix(self, roll, pitch, yaw):
        # convert angles to radians
        roll = np.radians(roll)
        pitch = np.radians(pitch)
        yaw = np.radians(yaw)

        # Rotation around X-axis (Roll)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll),  np.cos(roll)]
        ])
        
        # Rotation around Y-axis (Pitch)
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        # Rotation around Z-axis (Yaw)
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw),  np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation matrix: R = Rz * Ry * Rx
        R = Rz @ Ry @ Rx
        return R
    
if __name__ == "__main__":
    transfroms = TartuRawTransfroms("2024-03-25-15-40-16_mapping_tartu.csv")
    print(transfroms.transfroms[0])
