#!/usr/bin/env python3
from argparse import ArgumentParser
from functools import lru_cache
from pathlib import Path
from pypcd4 import PointCloud

import os
import numpy as np
from tqdm import tqdm

from liso.jcp.jcp import JPCGroundRemove
from tartu_raw_transfroms import TartuRawTransfroms


@lru_cache(maxsize=32)
def load_tartu_pcl_image_projection_get_ground_label(pcd_file: str):
    tartu_pcd = PointCloud.from_path(pcd_file).numpy()[:, :4]
    tartu_pcd[:, 3] /= 255
    is_ground = JPCGroundRemove(
        pcl=tartu_pcd[:, :3],
        range_img_width=1024,
        range_img_height=32,
        sensor_height=2.11,
        delta_R=1,
    )
    return tartu_pcd, is_ground


def get_timestamps(points):
        x = points[:, 0]
        y = points[:, 1]
        yaw = -np.arctan2(y, x)
        timestamps = 0.5 * (yaw / np.pi + 1.0)
        return timestamps


def rpy_to_rotation_matrix(roll, pitch, yaw):
    # convert angles to radians
    roll = np.radians(roll)
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)

    # Rotation around X-axis (Roll)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])
    
    # Rotation around Y-axis (Pitch)
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    # Rotation around Z-axis (Yaw)
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation matrix: R = Rz * Ry * Rx
    R = Rz @ Ry @ Rx
    return R


def main():
    from kiss_icp.config import KISSConfig
    from kiss_icp.kiss_icp import KissICP


    argparser = ArgumentParser(description="Convert tartu raw data to training format.")
    argparser.add_argument(
        "--target_dir",
        required=True,
        type=Path,
    )
    argparser.add_argument(
        "--filtered_data_file",
        required=True,
        type=Path,
    )
    argparser.add_argument(
        "--tartu_raw_root",
        required=True,
        type=Path,
    )
    argparser.add_argument(
        "--pcd_locations",
        required=True,
        type=Path,
    )
    args = argparser.parse_args()

    target_dir = args.target_dir / "tartu_filtered/train"

    with open(args.filtered_data_file) as file:
        lines = file.readlines()

    pcd_location_root = Path(args.pcd_locations)

    bl_T_velo_rotation_matrix = rpy_to_rotation_matrix(-0.0050194, -0.0047068, -0.055)
    bl_T_velo_translation_matrix = np.array([1.02, 0, 1.51]).reshape(3, 1)
    bl_T_velo = np.vstack((np.hstack((bl_T_velo_rotation_matrix, bl_T_velo_translation_matrix)), [0, 0, 0, 1]))

    skipped_sequences = 0
    success = 0
    date_count = 0
    last_date = ""
    for line in tqdm(lines):
        line_elems = line.strip().split(" ")
        date = line_elems[0]
        start = int(line_elems[1])
        end = int(line_elems[2])

        if last_date == date:
            date_count += 1
        
        last_date = date

        if not os.path.isdir(Path(args.tartu_raw_root) / date):
            print("Not a directory:", Path(args.tartu_raw_root) / date)
            continue
        
        lidar_data = Path(args.tartu_raw_root) / date / "lidar_center"
        csv_file = pcd_location_root / f"{date}.csv"
        if not os.path.isfile(csv_file):
            print(csv_file)
            continue

        tartu_transfroms = TartuRawTransfroms(csv_file)

        kiss_config = KISSConfig()
        kiss_config.mapping.voxel_size = 0.01 * kiss_config.data.max_range
        odometry = KissICP(config=kiss_config)

        seq_idxs = [i for i in range(start, end+1)]

        fnames = []

        for seq_idx in tqdm(seq_idxs[:-2], leave=False):

            if seq_idx not in tartu_transfroms.transfroms or seq_idx + 1 not in tartu_transfroms.transfroms or seq_idx + 2 not in tartu_transfroms.transfroms:
                 skipped_sequences += 1
                 continue
            
            t0_pcd_file = str(seq_idx).zfill(6) + ".pcd"
            pcl_t0, is_ground_t0 = load_tartu_pcl_image_projection_get_ground_label(lidar_data / t0_pcd_file)

            timestamps = get_timestamps(pcl_t0).astype(np.float64)
            odometry.register_frame(np.copy(pcl_t0[:, :3]).astype(np.float64), timestamps=timestamps)
            t1_pcd_file = str(seq_idx+1).zfill(6) + ".pcd"
            t2_pcd_file = str(seq_idx+2).zfill(6) + ".pcd"
            pcl_t1, is_ground_t1 = load_tartu_pcl_image_projection_get_ground_label(lidar_data / t1_pcd_file)
            pcl_t2, is_ground_t2 = load_tartu_pcl_image_projection_get_ground_label(lidar_data / t2_pcd_file)

            map_T_bl_t0 = tartu_transfroms.transfroms[seq_idx]
            map_T_bl_t1 = tartu_transfroms.transfroms[seq_idx + 1]
            map_T_bl_t2 = tartu_transfroms.transfroms[seq_idx + 2]

            map_T_velo_t0 = map_T_bl_t0 @ bl_T_velo
            map_T_velo_t1 = map_T_bl_t1 @ bl_T_velo
            map_T_velo_t2 = map_T_bl_t2 @ bl_T_velo

            odom_t0_t1 = np.linalg.inv(map_T_velo_t0) @ map_T_velo_t1
            odom_t0_t2 = np.linalg.inv(map_T_velo_t0) @ map_T_velo_t2
            sample_name = f"{date}_{date_count}_{seq_idx}"
            data_dict = {
                "pcl_t0": pcl_t0.astype(np.float32),
                "pcl_t1": pcl_t1.astype(np.float32),
                "pcl_t2": pcl_t2.astype(np.float32),
                "is_ground_t0": is_ground_t0,
                "is_ground_t1": is_ground_t1,
                "is_ground_t2": is_ground_t2,
                "odom_t0_t1": odom_t0_t1.astype(np.float64),
                "odom_t0_t2": odom_t0_t2.astype(np.float64),
                "name": sample_name,
            }

            target_fname = target_dir / Path(sample_name)
            fnames.append(target_fname)
            np.save(target_fname, data_dict,)

            if seq_idx == seq_idxs[-3]:
                timestamps = get_timestamps(pcl_t1).astype(np.float64)
                odometry.register_frame(np.copy(pcl_t1[:, :3]).astype(np.float64), timestamps=timestamps,)
                timestamps = get_timestamps(pcl_t2).astype(np.float64)
                odometry.register_frame(np.copy(pcl_t2[:, :3]).astype(np.float64), timestamps=timestamps,)

            success += 1
        w_Ts_si = odometry.poses

        for file_idx, fname in enumerate(fnames):
            if len(w_Ts_si) <= file_idx + 2:
                print(fname)
                continue
            content = np.load(fname.with_suffix(".npy"), allow_pickle=True).item()
            kiss_odom_t0_t1 = (np.linalg.inv(w_Ts_si[file_idx]) @ w_Ts_si[file_idx + 1])
            kiss_odom_t0_t2 = (np.linalg.inv(w_Ts_si[file_idx]) @ w_Ts_si[file_idx + 2])
            kiss_odom_t1_t2 = (np.linalg.inv(w_Ts_si[file_idx + 1]) @ w_Ts_si[file_idx + 2])
            content["kiss_odom_t0_t1"] = kiss_odom_t0_t1
            content["kiss_odom_t1_t0"] = np.linalg.inv(kiss_odom_t0_t1)
            content["kiss_odom_t0_t2"] = kiss_odom_t0_t2
            content["kiss_odom_t2_t0"] = np.linalg.inv(kiss_odom_t0_t2)
            content["kiss_odom_t1_t2"] = kiss_odom_t1_t2
            content["kiss_odom_t2_t1"] = np.linalg.inv(kiss_odom_t1_t2)

            np.save(fname, content)

    print(f"Skipped: {skipped_sequences} Success: {success}")


if __name__ == "__main__":
    main()
