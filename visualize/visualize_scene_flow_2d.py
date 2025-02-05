
import os
import numpy as np
import matplotlib.pyplot as plt

input_dir = "slim_static_flow"

for file in os.listdir(input_dir):
    print(file)

    file_path = os.path.join(input_dir, file)

    #point_cloud_data = np.load(file_path, allow_pickle=True).item()
    data = np.load(file_path)
    #print(data)

    scene_flow = data['bev_raw_flow_t0_t1']

    # Grid dimensions
    grid_size = 920
    real_world_size = 120  # meters (corresponds to 120m x 120m)

    # Create mesh grid for visualization
    x = np.linspace(-real_world_size / 2, real_world_size / 2, grid_size)
    y = np.linspace(-real_world_size / 2, real_world_size / 2, grid_size)
    X, Y = np.meshgrid(x, y)

    # Extract flow vectors
    U = scene_flow[:, :, 0]  # X-direction flow
    V = scene_flow[:, :, 1]  # Y-direction flow

    # Downsample for better visualization
    step = 15  # Adjust for clarity
    X_down, Y_down = X[::step, ::step], Y[::step, ::step]
    U_down, V_down = U[::step, ::step], V[::step, ::step]

    # Plot using quiver
    plt.figure(figsize=(10, 10))
    plt.quiver(X_down, Y_down, U_down, V_down, angles='xy', scale_units='xy', scale=1, color='r')

    # Plot formatting
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.title("LiDAR Scene Flow Visualization")
    plt.xlim([-real_world_size/2, real_world_size/2])
    plt.ylim([-real_world_size/2, real_world_size/2])
    plt.grid()
    plt.show()

    break
