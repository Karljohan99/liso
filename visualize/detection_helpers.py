import math
import numpy as np

from dataclasses import dataclass, field

@ dataclass
class Vector3:
    x: float = 0.
    y: float = 0.
    z: float = 0.

@dataclass
class DetectedObject:
    id: int
    position: Vector3
    dimensions: Vector3
    heading: float
    map_dimensions: Vector3 = None
    map_heading: float = None
    convex_hull_points = []
    convex_hull_map_points = []
    velocity: Vector3 = field(default_factory = Vector3)
    acceleration: Vector3 = field(default_factory = Vector3)
    valid: bool = False
    velocity_reliable: bool = False
    acceleration_reliable: bool = False


def calculate_iou(boxes1, boxes2):
    """
    Calculate the IOU between two sets of bounding boxes.

    Args:
        boxes1: a numpy array of shape (n, 4) containing the coordinates of n bounding boxes in the format (x1, y1, x2, y2).
        boxes2: a numpy array of shape (m, 4) containing the coordinates of m bounding boxes in the format (x1, y1, x2, y2).

    Returns:
        a numpy array of shape (n, m) containing the IOU between all pairs of bounding boxes.
    """
    # Calculate the area of each bounding box
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Calculate the coordinates of the intersection bounding boxes
    intersection_x1 = np.maximum(boxes1[:, 0][:, np.newaxis], boxes2[:, 0])
    intersection_y1 = np.maximum(boxes1[:, 1][:, np.newaxis], boxes2[:, 1])
    intersection_x2 = np.minimum(boxes1[:, 2][:, np.newaxis], boxes2[:, 2])
    intersection_y2 = np.minimum(boxes1[:, 3][:, np.newaxis], boxes2[:, 3])

    # Calculate the area of the intersection bounding boxes
    intersection_area = np.maximum(intersection_x2 - intersection_x1, 0) * np.maximum(intersection_y2 - intersection_y1, 0)

    # Calculate the union of the bounding boxes
    union_area = area1[:, np.newaxis] + area2 - intersection_area

    # Calculate the IOU
    iou = intersection_area / union_area

    return iou

def update_object_position_dimensions(obj):
    """
    Update width, length and position of the object, based on object's velocity vector aligned bounding box
    :param obj: DetectedObject
    """

    # Collect points from convex_hull and extract rotation center
    points = np.array([(p.x, p.y) for p in obj.convex_hull_map_points])
    centroid = np.array([obj.map_position.x, obj.map_position.y])
    heading_angle = get_heading_from_vector(obj.velocity)

    # Create rotation matrix
    cos_angle = np.cos(-heading_angle)
    sin_angle = np.sin(-heading_angle)
    rotation_matrix = np.array([
        [cos_angle, -sin_angle],
        [sin_angle, cos_angle]
    ])

    # Translate and rotate points
    points -= centroid
    points = points @ rotation_matrix.T

    # Calculate bounds in the rotated coordinate system
    minx, miny = points.min(axis=0)
    maxx, maxy = points.max(axis=0)
    width = (maxy - miny)
    length = (maxx - minx)
    center_x = (minx + maxx) / 2
    center_y = (miny + maxy) / 2

    # bounding box center
    target_point = np.array([center_x, center_y])

    # Create inverse rotation matrix
    # sin(-a) = -sin(a), cos(-a) = cos(a)
    inverse_rotation_matrix = np.array([
        [cos_angle, sin_angle],
        [-sin_angle, cos_angle]
    ])

    # Apply inverse rotation to target points, then translation
    target_point = target_point @ inverse_rotation_matrix.T
    target_point += centroid

    obj.map_position.x = target_point[0]
    obj.map_position.y = target_point[1]
    obj.dimensions.x = length
    obj.dimensions.y = width
    obj.map_heading = heading_angle

def get_heading_from_vector(vector):
    """
    Get heading from vector
    :param vector: vector
    :return: heading in radians
    """

    return math.atan2(vector.y, vector.x)