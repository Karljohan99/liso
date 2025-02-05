import numpy as np
import open3d as o3d

#path = "processed_data/2011_09_26_0011_0000000150.npy"
path = "processed_data/2024-04-02-12-11-04_mapping_tartu_streets_0_210.npy"

# Load the .npy file
data = np.load(path, allow_pickle=True).item()

# Extract point clouds and odometry
pcd_t0 = o3d.geometry.PointCloud()
pcd_t0.points = o3d.utility.Vector3dVector(data["pcl_t0"][:, :3])  # Point cloud at t0

pcd_t1 = o3d.geometry.PointCloud()
pcd_t1.points = o3d.utility.Vector3dVector(data["pcl_t1"][:, :3])  # Point cloud at t1

# Extract odometry (4x4 transformation matrix)
T_t0_t1 = np.array(data["kiss_odom_t0_t1"])  # Odometry from t0 to t1

# Apply transformations
pcd_t1.transform(T_t0_t1)  # Transform t1 to match t0

# Assign colors for differentiation
pcd_t0.paint_uniform_color([1, 0, 0])  # Red
pcd_t1.paint_uniform_color([0, 1, 0])  # Green

# Visualize all together
o3d.visualization.draw_geometries([pcd_t0, pcd_t1])

