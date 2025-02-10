import argparse
import numpy as np
import open3d as o3d

def get_3d_boxes(boxes):
    """
    Get 3D bounding boxes in proper Open3D format.

    Args:
        boxes (dict): Dictionary containing bounding box parameters.

    Returns:
        out_boxes (list): List of Open3D 3D bounding boxes.
    """
    out_boxes = []

    # Iterate through bounding boxes and visualize them
    for i in range(len(boxes['pos'])):
        center = boxes['pos'][i]
        dims = boxes['dims'][i]
        rot = boxes['rot'][i]
        is_valid = boxes['valid'][i]

        color = [0, 0, 1] if is_valid else [1, 0, 0]  # Blue for valid, Red for invalid
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


def get_scene_elements(dataset, i):
    if dataset == "kitti":
        # Load LiDAR point cloud
        point_cloud_path = f"pcd_files/kitti/kitti13/0000000{i}.bin"
        point_cloud = np.fromfile(point_cloud_path, dtype=np.float32)
        point_cloud = point_cloud.reshape(-1, 4)  # Nx4 (intensity is included)
        
        # Create Open3D point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])

        # Load 3D bounding boxes
        tracking_data = np.load("kitti_mined_dbs/tracked.npz", allow_pickle=True)
        boxes = tracking_data["arr_0"].item()[f"2011_09_26_0013_0000000{i}"]["raw_box"]

    elif dataset == "tartu":
        # Load LiDAR point cloud
        point_cloud_path = f"pcd_files/tartu/000{i}.pcd"
        pcd = o3d.io.read_point_cloud(point_cloud_path)

        # Load 3D bounding boxes
        tracking_data = np.load("tartu_mined_dbs/tracked.npz", allow_pickle=True)
        boxes = tracking_data["arr_0"].item()[f"2024-04-02-12-11-04_mapping_tartu_streets_0_{i}"]["raw_box"]

    else:
        raise ValueError("Unknown dataset")

    gray_color = np.full((len(pcd.points), 3), 0.5)  # 0.5 = medium gray
    pcd.colors = o3d.utility.Vector3dVector(gray_color)

    
    out_boxes = get_3d_boxes(boxes)

    return pcd, out_boxes


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    #parser.add_argument("--sequence")
    parser.add_argument("--i")
    args = parser.parse_args()

    i = str(args.i).zfill(3)

    pcd, boxes = get_scene_elements(args.dataset, i)

    # Visualize all objects
    o3d.visualization.draw_geometries([pcd] + boxes)


