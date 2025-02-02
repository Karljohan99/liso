#!/bin/bash

conda init && source $HOME/.bashrc && conda activate liso

python $HOME/liso/liso/datasets/tartu/create_tartu_raw.py --tartu_raw_root /mnt/ml2024 --target_dir /mnt/Bolt/liso/tartu_dataset --pcd_locations /mnt/Bolt/liso/tartu_pcd_locations
