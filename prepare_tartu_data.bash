#!/bin/bash

conda init && source $HOME/.bashrc && conda activate liso

python liso/datasets/tartu/create_tartu_filtered.py --target_dir /mnt/Bolt/liso/tartu_dataset --filtered_data_file liso/datasets/tartu/data.txt --tartu_raw_root /mnt/ml2024 --pcd_locations /mnt/Bolt/liso/tartu_pcd_locations
