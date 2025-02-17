import argparse
import numpy as np
import open3d as o3d

from points_clusterer import PointsClusterer


def get_3d_boxes_from_dict(boxes):
    """
    Convert 3D bounding boxes from liso dict format to Open3D format.

    Args:
        boxes (dict): Dictionary containing bounding box parameters.

    Returns:
        out_boxes (list): List of Open3D 3D bounding boxes.
    """
    out_boxes = []

    for i in range(len(boxes['pos'])):
        center = boxes['pos'][i]
        dims = boxes['dims'][i]
        rot = boxes['rot'][i]
        is_valid = boxes['valid'][i]

        color = [0, 0, 1] if is_valid else [1, 0, 0]  # Blue for valid, Red for invalid
        box_3d = create_3d_bbox(center, dims, rot, color)
        out_boxes.append(box_3d)

    return out_boxes


def get_3d_boxes_from_detected_object(boxes):
    """
    Convert 3D bounding boxes from DetectedObject class to Open3D format.

    Args:
        boxes (list): List of DetectedObject classes containing bounding box parameters.

    Returns:
        out_boxes (list): List of Open3D 3D bounding boxes.
    """
    out_boxes = []

    for box in boxes:
        center = [box.position.x, box.position.y, box.position.z]
        dims = [box.dimensions.x, box.dimensions.y, box.dimensions.z]
        rot = box.heading
        is_valid = box.valid

        color = [0, 0, 1] if is_valid else [1, 0, 0]  # Blue for valid, Red for invalid
        box_3d = create_3d_bbox(center, dims, rot, color)
        out_boxes.append(box_3d)

    return out_boxes

def get_3d_boxes_from_list(boxes):
    """
    Convert 3D bounding boxes from list to Open3D format.

    Args:
        boxes (list): 2D list containing bounding box parameters.

    Returns:
        out_boxes (list): List of Open3D 3D bounding boxes.
    """
    out_boxes = []

    for box in boxes:
        line = box.strip().split(" ")
        center = [float(line[0]), float(line[1]), float(line[2])]
        dims = [float(line[3]), float(line[4]), float(line[5])]
        rot = float(line[6])

        color = [0, 0, 1]
        box_3d = create_3d_bbox(center, dims, rot, color)
        out_boxes.append(box_3d)

    return out_boxes
    

def create_3d_bbox(center, dims, rot, color):
    """
    Creates a 3D bounding box with rotation.

    Args:
        center (list): [x, y, z] coordinates of the box center.
        dims (list): [length, width, height] of the box.
        rot (float): Rotation around the Z-axis in radians.
        color (list): RGB color for the box.

    Returns:
        o3d.geometry.LineSet: Open3D 3D bounding box.
    """
    # Define box corners in local coordinates
    l, w, h = dims
    corners = np.array([
        [-l/2, -w/2, -h/2], [l/2, -w/2, -h/2], [l/2, w/2, -h/2], [-l/2, w/2, -h/2],
        [-l/2, -w/2, h/2],  [l/2, -w/2, h/2],  [l/2, w/2, h/2],  [-l/2, w/2, h/2]
    ])

    # Rotation matrix (Z-axis rotation)
    cos_theta, sin_theta = np.cos(rot), np.sin(rot)
    rot_mat = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta,  cos_theta, 0],
        [0,          0,         1]
    ])

    # Rotate and translate corners
    rotated_corners = (rot_mat @ corners.T).T + np.array(center)

    # Define edges of the bounding box
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # Connecting edges
    ]

    # Create Open3D LineSet for bounding box
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(rotated_corners)
    line_set.lines = o3d.utility.Vector2iVector(edges)
    line_set.colors = o3d.utility.Vector3dVector([color] * len(edges))  # Set color for all edges

    return line_set


def get_scene_elements(dataset, sequence, i, points_clusterer, tracking_data=None):
    if dataset == "kitti":
        # Load LiDAR point cloud
        point_cloud_path = f"pcd_files/kitti/{sequence}/0000000{i}.bin"
        point_cloud = np.fromfile(point_cloud_path, dtype=np.float32)
        point_cloud = point_cloud.reshape(-1, 4)  # Nx4 (intensity is included)
        
        # Create Open3D point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])

        if points_clusterer == "simple":
            # Use clusterer to detect 3D bounding boxes
            points_clusterer = PointsClusterer("axis_aligned", 0.7, 4, False)
            boxes = points_clusterer.cluster(point_cloud[:, :3])
            out_boxes = get_3d_boxes_from_detected_object(boxes)

        elif points_clusterer == "liso":
            # Load 3D bounding boxes
            if tracking_data is None:
                tracking_data = np.load("kitti_mined_dbs/tracked.npz", allow_pickle=True)
                boxes = tracking_data["arr_0"].item()[f"2011_09_26_0013_0000000{i}"]["raw_box"]
            else:
                boxes = tracking_data[f"{bag_name}{i}"]["raw_box"]
            out_boxes = get_3d_boxes_from_dict(boxes)
        else:
            raise ValueError

    elif dataset == "tartu":
        if sequence == "tartu2":
            pcd_i = "64"
            bag_name = "2024-04-12-15-53-45_mapping_tartu_streets_47_64"
        else:
            pcd_i = "00"
            bag_name = "2024-04-02-12-11-04_mapping_tartu_streets_0_"

        # Load LiDAR point cloud
        point_cloud_path = f"pcd_files/tartu/{sequence}/0{pcd_i}{i}.pcd"
        pcd = o3d.io.read_point_cloud(point_cloud_path)

        if points_clusterer == "simple":
            # Use clusterer to detect 3D bounding boxes
            points_clusterer = PointsClusterer("axis_aligned", 0.7, 4, False)
            boxes = points_clusterer.cluster(np.asarray(pcd.points))
            out_boxes = get_3d_boxes_from_detected_object(boxes)

        elif points_clusterer == "liso":
            # Load 3D bounding boxes
            if tracking_data is None:
                tracking_data = np.load("tartu_mined_dbs/tracked.npz", allow_pickle=True)
                boxes = tracking_data["arr_0"].item()[f"{bag_name}{i}"]["raw_box"]
            else:
                boxes = tracking_data[f"{bag_name}{i}"]["raw_box"]
            out_boxes = get_3d_boxes_from_dict(boxes)

        elif points_clusterer == "aw_mini":
            # Bounding boxes from AW Mini
            filepath = f"aw_mini_tracked_objs/{dataset}/{sequence}/0{pcd_i}{i}.txt"
            with open(filepath) as f:
                out_boxes = get_3d_boxes_from_list(f.readlines())

        else:
            raise ValueError

    else:
        raise ValueError("Unknown dataset")

    gray_color = np.full((len(pcd.points), 3), 0.5)  # 0.5 = medium gray
    pcd.colors = o3d.utility.Vector3dVector(gray_color)

    return pcd, out_boxes


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    #parser.add_argument("--sequence")
    parser.add_argument("--points_clusterer")
    parser.add_argument("--i")
    args = parser.parse_args()

    i = str(args.i).zfill(3)

    pcd, boxes = get_scene_elements(args.dataset, i, args.points_clusterer)

    # Visualize all objects
    o3d.visualization.draw_geometries([pcd] + boxes)


