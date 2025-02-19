#!/usr/bin/env python3

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from detection_helpers import calculate_iou, update_object_position_dimensions

class EMATracker:
    def __init__(self):

        # Parameters
        self.enable_initial_velocity_estimate = False
        self.enable_initial_acceleration_estimate = False
        self.enable_missed_detection_propagation = True
        self.detection_counter_threshold = 4
        self.missed_counter_threshold = 2
        self.iou_threshold = 0.0
        self.velocity_gain = 0.2
        self.acceleration_gain = 0.0
        self.association_method = "iou"  # iou, euclidean
        self.max_euclidean_distance = 2.0

        self.tracked_objects = []
        self.tracked_objects_array = np.empty((0,), dtype=[
            ('centroid', np.float32, (2,)),
            ('bbox', np.float32, (4,)),
            ('velocity', np.float32, (2,)),
            ('acceleration', np.float32, (2,)),
            ('missed_counter', np.int32),
            ('detection_counter', np.int32),
        ])
        self.track_id_counter = 0

    def track_objects(self, clusters, transform):
        ### 1. PREPARE DETECTIONS ###

        # convert detected objects into Numpy array
        detected_objects_array = np.empty((len(clusters)), dtype=self.tracked_objects_array.dtype)
        for i, obj in enumerate(clusters):
            detected_objects_array[i]['centroid'] = (obj.map_position.x, obj.map_position.y)
            detected_objects_array[i]['bbox'] = (obj.map_position.x - obj.dimensions.x / 2, obj.map_position.y - obj.dimensions.y / 2,
                                                 obj.map_position.x + obj.dimensions.x / 2, obj.map_position.y + obj.dimensions.y / 2)
            detected_objects_array[i]['velocity'] = (obj.velocity.x, obj.velocity.y)
            detected_objects_array[i]['acceleration'] = (obj.acceleration.x, obj.acceleration.y)
            detected_objects_array[i]['missed_counter'] = 0
            detected_objects_array[i]['detection_counter'] = 1
        assert len(clusters) == len(detected_objects_array)

        inverse_transform = np.linalg.inv(transform)

        ### 2. PROPAGATE EXISTING TRACKS FORWARD ###

        # difference between current and previous message
        time_delta = 0.1

        # move tracked objects forward in time
        assert len(self.tracked_objects) == len(self.tracked_objects_array), str(len(self.tracked_objects)) + ' ' + str(len(self.tracked_objects_array))
        position_change = time_delta * self.tracked_objects_array['velocity']
        tracked_object_centroids = self.tracked_objects_array['centroid'].copy()
        tracked_object_centroids += position_change
        tracked_object_bboxes = self.tracked_objects_array['bbox'].copy()
        tracked_object_bboxes[:,:2] += position_change
        tracked_object_bboxes[:,2:4] += position_change

        ### 3. MATCH TRACKS WITH DETECTIONS ###

        if self.association_method == 'iou':
            # Calculate the IOU between the tracked objects and the detected objects
            iou = calculate_iou(tracked_object_bboxes, detected_objects_array['bbox'])
            assert iou.shape == (len(self.tracked_objects_array), len(detected_objects_array)), str(iou.shape) + ' ' + str((len(self.tracked_objects_array), len(detected_objects_array)))

            # Calculate the association between the tracked objects and the detected objects
            matched_track_indices, matched_detection_indicies = linear_sum_assignment(-iou)
            assert len(matched_track_indices) == len(matched_detection_indicies)

            # Only keep those matches where the IOU is greater than threshold
            matches = iou[matched_track_indices, matched_detection_indicies] > self.iou_threshold
            matched_track_indices = matched_track_indices[matches]
            matched_detection_indicies = matched_detection_indicies[matches]
            assert len(matched_track_indices) == len(matched_detection_indicies)
        elif self.association_method == 'euclidean':
            # Calculate euclidean distance between the tracked object and the detected object centroids
            dists = cdist(tracked_object_centroids, detected_objects_array['centroid'])
            assert dists.shape == (len(self.tracked_objects_array), len(detected_objects_array))

            # Calculate the association between the tracked objects and the detected objects
            matched_track_indices, matched_detection_indicies = linear_sum_assignment(dists)

            # Only keep those matches where the distance is less than threshold
            matches = dists[matched_track_indices, matched_detection_indicies] <= self.max_euclidean_distance
            matched_track_indices = matched_track_indices[matches]
            matched_detection_indicies = matched_detection_indicies[matches]
            assert len(matched_track_indices) == len(matched_detection_indicies)
        else:
            assert False, 'Unknown association method: ' + self.association_method

        ### 4. ESTIMATE TRACKED OBJECT SPEEDS AND ACCELERATIONS ###

        # update tracked object speeds with exponential moving average
        new_velocities = (detected_objects_array['centroid'][matched_detection_indicies] - self.tracked_objects_array['centroid'][matched_track_indices]) / time_delta
        old_velocities = self.tracked_objects_array['velocity'][matched_track_indices]
        if self.enable_initial_velocity_estimate:
            # make initial velocity of an object equal to its first velocity estimate instead of zero
            second_time_detections = self.tracked_objects_array['detection_counter'][matched_track_indices] == 1
            old_velocities[second_time_detections] = new_velocities[second_time_detections]
        detected_objects_array['velocity'][matched_detection_indicies] = (1 - self.velocity_gain) * old_velocities + self.velocity_gain * new_velocities

        # update tracked object accelerations with exponential moving average
        new_accelerations = (new_velocities - old_velocities) / time_delta
        old_accelerations = self.tracked_objects_array['acceleration'][matched_track_indices]
        if self.enable_initial_acceleration_estimate:
            # make initial acceleration of an object equal to its first acceleration estimate instead of zero
            third_time_detections = self.tracked_objects_array['detection_counter'][matched_track_indices] == 2
            old_accelerations[third_time_detections] = new_accelerations[third_time_detections]
        detected_objects_array['acceleration'][matched_detection_indicies] = (1 - self.acceleration_gain) * old_accelerations + self.acceleration_gain * new_accelerations

        ### 5. UPDATE TRACKED OBJECTS ###

        # Replace tracked objects with detected objects, keeping the same ID
        for track_idx, detection_idx in zip(matched_track_indices, matched_detection_indicies):
            tracked_obj = self.tracked_objects[track_idx]
            detected_obj = clusters[detection_idx]
            detected_obj.id = tracked_obj.id
            if not detected_obj.velocity_reliable:
                detected_obj.velocity.x, detected_obj.velocity.y = detected_objects_array['velocity'][detection_idx]
                detected_obj.velocity_reliable = True
            if not detected_obj.acceleration_reliable:
                detected_obj.acceleration.x, detected_obj.acceleration.y = detected_objects_array['acceleration'][detection_idx]
                detected_obj.acceleration_reliable = True
            self.tracked_objects[track_idx] = detected_obj
        self.tracked_objects_array[['centroid', 'bbox', 'velocity', 'acceleration']][matched_track_indices] = \
            detected_objects_array[['centroid', 'bbox', 'velocity', 'acceleration']][matched_detection_indicies]

        ### 6. MANAGE TRACK STATUS ###

        # create missed track indices
        all_track_indices = np.arange(0, len(self.tracked_objects), 1, dtype=int)
        missed_track_indices = np.delete(all_track_indices, matched_track_indices)

        # create new detection indices
        all_detection_indices = np.arange(0, len(clusters), 1, dtype=int)
        new_detection_indices = np.delete(all_detection_indices, matched_detection_indicies)

        # zero missed counter and increase detected counter for matched objects
        self.tracked_objects_array['missed_counter'][matched_track_indices] = 0
        self.tracked_objects_array['detection_counter'][matched_track_indices] += 1

        # increase missed counter and zero detection counter for non-matched tracks
        self.tracked_objects_array['missed_counter'][missed_track_indices] += 1
        #self.tracked_objects_array['detection_counter'][missed_track_indices] = 0

        # move missed tracks forward
        if self.enable_missed_detection_propagation:
            assert len(self.tracked_objects) == len(self.tracked_objects_array) == len(tracked_object_centroids) == len(tracked_object_bboxes)
            self.tracked_objects_array['centroid'][missed_track_indices] = tracked_object_centroids[missed_track_indices]
            self.tracked_objects_array['bbox'][missed_track_indices] = tracked_object_bboxes[missed_track_indices]
            for idx in missed_track_indices:
                obj = self.tracked_objects[idx]
                obj.map_position.x, obj.map_position.y = self.tracked_objects_array['centroid'][idx]
                for p in obj.convex_hull_map_points:
                    p.x += position_change[idx][0]
                    p.y += position_change[idx][1]

        # delete stale tracks
        stale_track_indices = np.where(self.tracked_objects_array['missed_counter'] >= self.missed_counter_threshold)[0]
        for idx in sorted(stale_track_indices, reverse=True):
            del self.tracked_objects[idx]
        self.tracked_objects_array = np.delete(self.tracked_objects_array, stale_track_indices, axis=0)
        assert len(self.tracked_objects) == len(self.tracked_objects_array), str(len(self.tracked_objects)) + ' ' + str(len(self.tracked_objects_array))

        # add new detections
        for obj_idx in new_detection_indices:
            detected_obj = clusters[obj_idx]
            detected_obj.id = self.track_id_counter
            self.track_id_counter += 1
            self.tracked_objects.append(detected_obj)
        self.tracked_objects_array = np.concatenate((self.tracked_objects_array, detected_objects_array[new_detection_indices]))
        assert len(self.tracked_objects) == len(self.tracked_objects_array)

        ### 7. PUBLISH CURRENT TRACKS ###

        # filter out objects with enough detections
        tracked_objects_indices = np.where(self.tracked_objects_array['detection_counter'] >= self.detection_counter_threshold)[0]
        tracked_objects = [self.tracked_objects[idx] for idx in tracked_objects_indices]
        assert len(tracked_objects) == len(tracked_objects_indices)

        ### 8. update tracked objects dimensions, position, heading based on velocity vector ###
        for i, obj in enumerate(tracked_objects):
            update_object_position_dimensions(obj)

            new_pos = inverse_transform @ np.array([obj.map_position.x, obj.map_position.y, 0, 1])
            obj.position.x = new_pos[0]
            obj.position.y = new_pos[1]
            
            new_v = inverse_transform[:3, :3] @ np.array([np.cos(obj.map_heading), np.sin(obj.map_heading), 0])
            obj.heading = np.arctan2(new_v[1], new_v[0])

        return tracked_objects