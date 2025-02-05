import numpy as np
import open3d as o3d
import argparse


def create_arrow(start, end, scale=0.8):
    """
    Creates an arrow from the center to the heading direction.
    
    Args:
        center (tuple): (x, y, z) center of bounding box.
        heading (tuple): (x, y, z) heading direction.
        scale (float): Scale of the arrow.

    Returns:
        o3d.geometry.TriangleMesh: 3D arrow mesh.
    """
    dx, dy, _ = end - start
    yaw = np.arctan2(dy, dx)  # Compute angle in radians

    magnitude = np.linalg.norm(np.array([dx, dy]))
    if magnitude <= 0.01:
        return None
    
    scale = magnitude*scale

    # Create an arrow mesh
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.05 * scale, 
        cone_radius=0.1 * scale, 
        cylinder_height=0.6 * scale, 
        cone_height=0.4 * scale
    )

    # Translate arrow to the center
    arrow.translate(start)
    
    R = o3d.geometry.get_rotation_matrix_from_zyx((0, np.pi / 2, -yaw)) # Arrow pointed towards z-axis at initialization
    arrow.rotate(R)

    # Set arrow color
    arrow.paint_uniform_color([0, 0, 1])  # Blue color for heading arrow

    """
    if abs(yaw) < np.pi/2 or abs(yaw) > 3*np.pi/2:
        arrow.paint_uniform_color([1, 0, 0])  # Red color for heading arrow
    else:
        arrow.paint_uniform_color([0, 0, 1])  # Blue color for heading arrow
    """
    return arrow

def visualize_scene_flow(scene_flow):
    """
    Visualizes LiDAR 2D scene flow vectors in Open3D.
    
    Args:
        scene_flow (np.ndarray): Scene flow array of shape (920, 920, 2).
    """

    # Define scene parameters
    grid_size = 920
    real_world_size = 120  # meters (120m x 120m)

    # Create a 2D grid that maps to LiDAR coordinates
    x = np.linspace(-real_world_size / 2, real_world_size / 2, grid_size)
    y = np.linspace(-real_world_size / 2, real_world_size / 2, grid_size)

    X, Y = np.meshgrid(x, y)
    U, V = scene_flow[:, :, 0], scene_flow[:, :, 1]
    Z = np.zeros_like(X)

    arrows = []
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            start = np.array([Y[i, j], X[i, j], Z[i, j]]) # Start of arrow
            end = start + np.array([U[i, j], V[i, j], 0]) # End of arrow

            arrow = create_arrow(start, end)
            if arrow is None:
                continue

            arrows.append(arrow)

    return arrows

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    parser.add_argument("--i")
    args = parser.parse_args()

    if args.data == "kitti":
        point_cloud_path = f"pcd_files/00000001{args.i}.bin" 
        pcd = o3d.geometry.PointCloud()
        point_cloud = np.fromfile(point_cloud_path, dtype=np.float32)  # Load raw data
        point_cloud = point_cloud.reshape(-1, 4)
        pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
        
        slim_pred_path = f"slim_static_flow/2011_09_26_0011_00000001{args.i}.npz"
    elif args.data == "tartu":
        point_cloud_path = f"pcd_files/0002{int(args.i)}.pcd" 
        pcd = o3d.io.read_point_cloud(point_cloud_path)
        
        slim_pred_path = f"slim_static_flow/2024-04-02-12-11-04_mapping_tartu_streets_0_2{args.i}.npz"
    else:
        raise ValueError
    
    gray_color = np.full((len(pcd.points), 3), 0.5)  # 0.5 = medium gray
    pcd.colors = o3d.utility.Vector3dVector(gray_color)

    data = np.load(slim_pred_path)
    scene_flow_data = data['bev_raw_flow_t0_t1']

    arrows = visualize_scene_flow(scene_flow_data)

    o3d.visualization.draw_geometries([pcd] + arrows)

