import numpy as np
import open3d as o3d


def transform_arrow(start, end, arrows, i, scale=0.8):
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
    if magnitude <= 0:
        return False
    
    scale = magnitude*scale
    arrows[i].scale(scale, center=a.get_center())

    # Translate arrow to the center
    arrows[i].translate(start)
    
    R = o3d.geometry.get_rotation_matrix_from_zyx((0, - np.pi / 2, -yaw)) # Arrow pointed towards z-axis at initialization
    arrows[i].rotate(R)

    # Set arrow color

    if abs(yaw) < np.pi/2 or abs(yaw) > 3*np.pi/2:
        arrows[i].paint_uniform_color([1, 0, 0])  # Red color for heading arrow
    else:
        arrows[i].paint_uniform_color([0, 0, 1])  # Blue color for heading arrow

    return True

def visualize_scene_flow(scene_flow, arrows):
    """
    Visualizes LiDAR 2D scene flow vectors in Open3D.
    
    Args:
        scene_flow (np.ndarray): Scene flow array of shape (920, 920, 2).
    """

    for a in arrows:
        a.translate(-a.get_center(), relative=True)
        a.translate([0, 0, 0], relative=False)
        bbox = a.get_axis_aligned_bounding_box()  
        scale_factor = 1.0 / max(bbox.get_extent())  # Normalize size  
        a.scale(scale_factor, center=a.get_center()) 

    # Define scene parameters
    grid_size = 920
    real_world_size = 120  # meters (120m x 120m)

    # Create a 2D grid that maps to LiDAR coordinates
    x = np.linspace(-real_world_size / 2, real_world_size / 2, grid_size)
    y = np.linspace(-real_world_size / 2, real_world_size / 2, grid_size)

    X, Y = np.meshgrid(x, y)
    U, V = scene_flow[:, :, 0], scene_flow[:, :, 1]
    Z = np.zeros_like(X)

    count = 0
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            start = np.array([Y[i, j], X[i, j], Z[i, j]]) # Start of arrow
            end = start + np.array([U[i, j], V[i, j], 0]) # End of arrow

            if transform_arrow(start, end, arrows, count):
                count += 1


vis = o3d.visualization.Visualizer()
vis.create_window()

pcd = o3d.geometry.PointCloud()
points = np.random.rand(100, 3)  # 100 points
pcd.points = o3d.utility.Vector3dVector(points)
vis.add_geometry(pcd)

arrows = []
for _ in range(10000):
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.05, 
        cone_radius=0.1, 
        cylinder_height=0.6, 
        cone_height=0.4
    )
    arrows.append(arrow)

for a in arrows:
    vis.add_geometry(a)

for i in range(9):

    point_cloud_path = f"pcd_files/00349{i}.pcd" 
    new_pcd = o3d.io.read_point_cloud(point_cloud_path)
    pcd.points = new_pcd.points
    gray_color = np.full((len(pcd.points), 3), 0.5)  # 0.5 = medium gray
    pcd.colors = o3d.utility.Vector3dVector(gray_color)

    vis.update_geometry(pcd)

    data = np.load(f"slim_static_flow/2024-04-02-11-46-51_mapping_tartu_streets_349{i}.npz")
    scene_flow_data = data['bev_raw_flow_t0_t1']

    view_control = vis.get_view_control()
    view_control.set_zoom(30)

    vis.poll_events()
    vis.update_renderer()

    visualize_scene_flow(scene_flow_data, arrows)

    for a in arrows:
        vis.update_geometry(a)
    
    #o3d.visualization.draw_geometries(geometries)

    #frame_path = f"frames/frame_{i:03d}.png"
    #o3d.io.write_image(frame_path, o3d.geometry.Image(np.asarray(o3d.visualization.render_to_image())))

    #time.sleep(0.5)

vis.destroy_window()
