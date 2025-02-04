#!/bin/bash

conda init && source $HOME/.bashrc && conda activate liso

python $HOME/liso/liso/datasets/tartu/python create_tartu_filtered.py --target_dir /mnt/Bolt/liso/tartu_dataset --filtered_data_file data.txt --tartu_raw_root /mnt/ml2024 --pcd_locations /mnt/Bolt/liso/tartu_pcd_locations
