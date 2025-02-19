import os
import numpy as np
import pandas as pd
import open3d as o3d

from points_clusterer import PointsClusterer
from road_area_filter import RoadAreaFilter
from ema_tracker import EMATracker


def track_sequence(dataroot, locations_name, results_path, start, end):
    points_clusterer = PointsClusterer("axis_aligned", 0.7, 4, False)
    road_area_filter = RoadAreaFilter("tartu_large.geojson")
    tracker = EMATracker()

    location_df = pd.read_csv(locations_name)

    for idx in range(start, end+1):
        pcd_idx = str(idx).zfill(6)

        # Load the PCD file
        pcd = o3d.io.read_point_cloud(os.path.join(dataroot, f"{pcd_idx}.pcd"))

        # Convert to NumPy array
        points = np.asarray(pcd.points)

        clustered_points = points_clusterer.cluster(points)

        filtered_objects, T = road_area_filter.filter_objects(clustered_points, location_df.loc[location_df["sequence"] == idx])

        tracked_objects = tracker.track_objects(filtered_objects, T)

        save_tracked_objects(os.path.join(results_path, f"{pcd_idx}.txt"), tracked_objects)

def save_tracked_objects(filename, tracked_objects):
    lines = []
    for i, obj in enumerate(tracked_objects):
        line = f"{obj.position.x} {obj.position.y} {obj.position.z} {obj.dimensions.x} {obj.dimensions.y} {obj.dimensions.z} {obj.heading}"
        if i != len(tracked_objects) - 1:
            line += "\n"
        lines.append(line)

    with open(filename, 'w') as file:
        file.writelines(lines)
        
if __name__ == '__main__':
    track_sequence("/home/pilve/liso/visualize/pcd_files/tartu/tartu1", "2024-04-02-12-11-04_mapping_tartu_streets.csv", "aw_mini_tracked_objs/tartu/tartu1", 200, 299)
    track_sequence("/home/pilve/liso/visualize/pcd_files/tartu/tartu2", "2024-04-12-15-53-45_mapping_tartu_streets.csv", "aw_mini_tracked_objs/tartu/tartu2", 64300, 64599)