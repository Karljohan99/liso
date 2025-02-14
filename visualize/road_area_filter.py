#!/usr/bin/env python3

import json
import numpy as np
import shapely

from pyproj import Transformer

class RoadAreaFilter:
    def __init__(self, road_area_file, map_extraction_distance=150):
        
        self.map_extraction_distance = map_extraction_distance

        origin_lat = 58.385345
        origin_lon = 26.726272

        # Set up a transformer for WGS84 to UTM
        self.wgs2utm_transformer = Transformer.from_crs("epsg:4326", "epsg:32635")  # UTM Zone 35N
        self.origin_easting, self.origin_northing = self.wgs2utm_transformer.transform(origin_lat, origin_lon)

        self.central_meridian = (origin_lon // 6.0) * 6.0 + 3.0

        # Read the GeoJSON file and create shapely geometries
        road_area_data = []
        with open(road_area_file, 'r') as f:
            self.geojson_data = json.load(f)
            for feature in self.geojson_data['features']:
                geometry = shapely.geometry.shape(feature['geometry'])
                road_area_data.append(geometry)

        road_area_data = shapely.unary_union(road_area_data)
        road_area_data = shapely.affinity.translate(road_area_data, xoff=-self.origin_easting, yoff=-self.origin_northing)
        shapely.prepare(road_area_data)
        self.road_area_data = road_area_data

    def correct_azimuth(self, lat, lon, azimuth):
        # calculate grid convergence and use to correct the azimuth
        # https://gis.stackexchange.com/questions/115531/calculating-grid-convergence-true-north-to-grid-north
        a = np.tan(np.radians(lon - self.central_meridian))
        b = np.sin(np.radians(lat))
        correction = np.degrees(np.arctan(a * b))

        # TODO compare with CA = (Lambda - LambdaCM) * sin Theta

        return azimuth - correction
            
    def filter_objects(self, detected_objects, location):

        _, _, longitude, latitude, height, azimuth, roll, pitch = location.iloc[0].tolist()

        x, y = self.wgs2utm_transformer.transform(latitude, longitude)
        x -= self.origin_easting
        y -= self.origin_northing

        map_base_link_T = self.create_bl_to_map_matrix(x, y, latitude, longitude, 0, azimuth, roll, pitch)

        map_extent_box = shapely.box(x - self.map_extraction_distance, 
                                     y - self.map_extraction_distance, 
                                     x + self.map_extraction_distance, 
                                     y + self.map_extraction_distance)
        
        road_area = map_extent_box.intersection(self.road_area_data)
        shapely.prepare(road_area)

        filtered_objects = []
        for obj in detected_objects:
            map_coords = map_base_link_T @ np.array([obj.position.x, obj.position.y, 0, 1])
            obj_geom = shapely.Point(map_coords[0], map_coords[1])

            if road_area.intersects(obj_geom):
                filtered_objects.append(obj)

        return filtered_objects
    

    def create_bl_to_map_matrix(self, x, y, longitude, latitude, height, azimuth, roll, pitch):
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
    
    def convertAzimuthToENU(self, roll, pitch, yaw):

        # These transforms are taken from gpsins_localizer_nodelet.cpp
        # Convert from Azimuth (CW from North) to ENU (CCW from East)
        yaw = -yaw + np.pi/2

        # Clamp within 0 to 2 pi
        if yaw > 2 * np.pi:
            yaw = yaw - 2 * np.pi
        elif yaw < 0:
            yaw += 2 * np.pi
        
        # Novatel GPS uses different vehicle body frame (y forward, x right, z up)
        pitch = -pitch

        return roll, pitch, yaw

    def rpy_to_rotation_matrix(self, roll, pitch, yaw):
        # convert angles to radians
        roll = np.radians(roll)
        pitch = np.radians(pitch)
        yaw = np.radians(yaw)

        roll, pitch, yaw = self.convertAzimuthToENU(roll, pitch, yaw)

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

