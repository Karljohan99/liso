#!/usr/bin/env python3

import numpy as np
import open3d as o3d

from cluster_detector import ClusterDetector
from naive_ground_removal import NaiveGroundRemoval

class PointsClusterer:
    def __init__(self, bounding_box_type, epsilon, min_size, in_2d):
        self.cluster_epsilon = epsilon
        self.cluster_min_size = min_size
        self.cluster_in_2d = in_2d

        try:
            from cuml.cluster import DBSCAN
            self.clusterer = DBSCAN(eps=self.cluster_epsilon, min_samples=self.cluster_min_size)
            print("Using DBSCAN from cuML")
        except ImportError:
            try:
                from sklearnex.cluster import DBSCAN
                self.clusterer = DBSCAN(eps=self.cluster_epsilon, min_samples=self.cluster_min_size, algorithm='auto')
                print("Using DBSCAN from Intel® Extension for Scikit-learn")
            except ImportError:
                from sklearn.cluster import DBSCAN
                self.clusterer = DBSCAN(eps=self.cluster_epsilon, min_samples=self.cluster_min_size, algorithm='ball_tree')
                print("Using DBSCAN from Scikit-learn")

        self.cluster_detector = ClusterDetector(min_size, bounding_box_type)
        self.naive_ground_removal = NaiveGroundRemoval()

    def cluster(self, raw_points):

        points = self.naive_ground_removal.remove_ground(raw_points)

        # get labels for clusters
        labels = self.clusterer.fit_predict(points[:, :2] if self.cluster_in_2d else points)

        # concatenate points with labels
        points_labeled = np.hstack((points, labels.reshape(-1, 1)))

        # filter out noise points
        points_labeled = points_labeled[labels != -1]

        boxes = self.cluster_detector.detect(points_labeled)

        return boxes

if __name__ == '__main__':
    # Load the PCD file
    pcd = o3d.io.read_point_cloud("pcd_files/tartu/000210.pcd")

    # Convert to NumPy array
    points = np.asarray(pcd.points)

    points_clusterer = PointsClusterer("axis_aligned", 0.7, 4, False)
    points_clusterer.cluster(points)
