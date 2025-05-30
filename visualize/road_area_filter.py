#!/usr/bin/env python3

import json
import numpy as np
import shapely
from pyproj import Transformer
from detection_helpers import Vector3

import matplotlib.pyplot as plt

class RoadAreaFilter:
    def __init__(self, road_area_file, map_extraction_distance=150):
        
        self.map_extraction_distance = map_extraction_distance
        self.filtering_method = "within" #centroid, intersects

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

        base_link_to_map_T = self.create_bl_to_map_matrix(x, y, latitude, longitude, 0, azimuth, roll, pitch)

        map_extent_box = shapely.box(x - self.map_extraction_distance, 
                                     y - self.map_extraction_distance, 
                                     x + self.map_extraction_distance, 
                                     y + self.map_extraction_distance)
        
        road_area = map_extent_box.intersection(self.road_area_data)
        shapely.prepare(road_area)

        if self.filtering_method == "within":
            not_road_area = map_extent_box.difference(road_area)
            shapely.prepare(not_road_area)

        filtered_objects = []
        points_list = []
        for obj in detected_objects:
            map_coords = base_link_to_map_T @ np.array([obj.position.x, obj.position.y, 0, 1])
            
            obj.map_position = Vector3(map_coords[0], map_coords[1], 0)
            v_new = base_link_to_map_T[:3, :3] @ np.array([np.cos(obj.heading), np.sin(obj.heading), 0])
            obj.map_heading = np.arctan2(v_new[1], v_new[0])

            map_convex_hull_points = []
            for p in obj.convex_hull_points:
                map_p = base_link_to_map_T @ np.array([p.x, p.y, p.z, 1])
                map_convex_hull_points.append(Vector3(map_p[0], map_p[1], 0))

            obj.convex_hull_map_points = map_convex_hull_points


            if self.filtering_method == "centroid":
                obj_geom = shapely.Point(map_coords[0], map_coords[1])
                points_list.append(obj_geom)
            else:
                obj_geom = shapely.polygons([(p.x, p.y) for p in obj.convex_hull_map_points])


            if self.filtering_method == "centroid" or self.filtering_method == "intersects":
                if road_area.intersects(obj_geom):
                    filtered_objects.append(obj)
            elif self.filtering_method == "within":
                if not not_road_area.intersects(obj_geom):
                    filtered_objects.append(obj)

        #visualize(road_area, points_list, (x, y))

        return filtered_objects, base_link_to_map_T
    

    def create_bl_to_map_matrix(self, x, y, longitude, latitude, height, azimuth, roll, pitch):
        #corrected_azimuth = self.correct_azimuth(latitude, longitude, azimuth)
        rotation_matrix = self.rpy_to_rotation_matrix(roll, pitch, azimuth)
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
    

def visualize(multi_poly, points, location):
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(6, 6))

    x, y = multi_poly.geoms[0].exterior.xy  # Get exterior coordinates
    ax.plot(x, y, color="blue", linewidth=2)  # Plot polygon edges

    # Plot points
    x_coords = [point.x for point in points]
    y_coords = [point.y for point in points]
    ax.scatter(x_coords, y_coords, color="red", marker="o", label="Boxes")
    ax.scatter([location[0]], [location[1]], color="green", marker="o", label="Location")

    # Set axis limits
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.legend()
    ax.set_aspect("equal")

    # Show plot
    plt.show()
