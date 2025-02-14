#!/usr/bin/env python3

import math
import numpy as np
import cv2

class NaiveGroundRemoval:
    def __init__(self):
        self.min_x = -60.0
        self.max_x = 60.0
        self.min_y = -60.0
        self.max_y = 60.0
        self.min_z = -2.5
        self.max_z =  0.05
        self.cell_size = 0.6
        self.tolerance = 0.15
        self.filter = "average"
        self.filter_size = 3
        self.filter_iterations = 1

        self.width = int(math.ceil((self.max_x - self.min_x) / self.cell_size))
        self.height = int(math.ceil((self.max_y - self.min_y) / self.cell_size))
        self.cols = np.empty((self.width, self.height), dtype=np.float32)

    def remove_ground(self, points):
        
        # filter out of range points
        filter = (self.min_x <= points[:, 0]) & (points[:, 0] < self.max_x) \
               & (self.min_y <= points[:, 1]) & (points[:, 1] < self.max_y) \
               & (self.min_z <= points[:, 2]) & (points[:, 2] < self.max_z)
        points_filtered = points[filter]

        # convert x and y coordinates into indexes
        xi = ((points_filtered[:, 0] - self.min_x) / self.cell_size).astype(np.int32)
        yi = ((points_filtered[:, 1] - self.min_y) / self.cell_size).astype(np.int32)
        zi = points_filtered[:, 2]

        # write minimum height for each cell to cols
        # thanks to sorting in descending order,
        # the minimum value will overwrite previous values
        self.cols.fill(np.nan)
        idx = np.argsort(-zi)
        self.cols[xi[idx], yi[idx]] = zi[idx]

        # bring cell minimum lower, if all cells around it are lower
        for _ in range(self.filter_iterations):
            if self.filter == 'median':
                cols_filtered = cv2.medianBlur(self.cols, self.filter_size)
                np.fmin(self.cols, cols_filtered, out=self.cols)
            elif self.filter == 'average':
                mask = np.isnan(self.cols)
                self.cols[mask] = 0
                cols_filtered = cv2.blur(self.cols, (self.filter_size, self.filter_size), cv2.BORDER_REPLICATE) / \
                        cv2.blur((~mask).astype(np.float32), (self.filter_size, self.filter_size), cv2.BORDER_REPLICATE)
                np.fmin(self.cols, cols_filtered, out=self.cols)
            elif self.filter == 'minimum':
                mask = np.isnan(self.cols)
                self.cols[mask] = np.inf
                cols_filtered = cv2.erode(self.cols, np.ones((self.filter_size, self.filter_size)), cv2.BORDER_REPLICATE)
                np.fmin(self.cols, cols_filtered, out=self.cols)
            elif self.filter != 'none':
                assert False, "Unknown filter value: " + self.filter

        # filter out closest points to minimum point up to some tolerance
        ground_mask = (zi <= (self.cols[xi, yi] + self.tolerance))

        return points_filtered[~ground_mask]