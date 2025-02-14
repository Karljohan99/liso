#!/usr/bin/env python3

import math
import numpy as np
import cv2

from detection_helpers import DetectedObject, Vector3

class ClusterDetector:
    def __init__(self, min_size, bounding_box_type):
        self.min_cluster_size = min_size
        self.bounding_box_type = bounding_box_type

    def detect(self, points):

        # make copy of labels
        labels = points[:, 3]
        labels = labels.astype(np.int64)

        if len(labels) == 0:
            num_clusters = 0
        else:
            num_clusters = np.max(labels) + 1

        clusters = []
        for i in range(num_clusters):
            # filter points for this cluster
            mask = (labels == i)

            # ignore clusters smaller than certain size
            if np.sum(mask) < self.min_cluster_size:
                continue

            # fetch points for this cluster
            points3d = points[mask,:3]
            # cv2.convexHull needs contiguous array of 2D points
            points2d = np.ascontiguousarray(points3d[:,:2])

            if self.bounding_box_type == 'axis_aligned':
                # calculate centroid and dimensions
                maxs = np.max(points3d, axis=0)
                mins = np.min(points3d, axis=0)
                center_x, center_y, center_z = np.mean(points3d, axis=0)
                dim_x, dim_y, dim_z = maxs - mins

                # always pointing forward
                heading = 0.0
            elif self.bounding_box_type == 'min_area':
                # calculate minimum area bounding box
                (center_x, center_y), (dim_x, dim_y), heading_angle = cv2.minAreaRect(points2d)

                # convert degrees to radians for heading angle
                heading = math.radians(heading_angle)

                # calculate height and vertical position
                max_z = np.max(points3d[:,2])
                min_z = np.min(points3d[:,2])
                dim_z = max_z - min_z
                center_z = (max_z + min_z) / 2.0
            else:
                raise ValueError("wrong bounding_box_type: " + self.bounding_box_type)

            cluster = DetectedObject(i, Vector3(center_x, center_y, center_z), Vector3(dim_x, dim_y, dim_z), heading)
            cluster.valid = True

            clusters.append(cluster)

        return clusters