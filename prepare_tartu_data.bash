#!/bin/bash

export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

ldd --version

source $HOME/liso/install_extra_packages.bash

conda init && source $HOME/.bashrc && conda activate liso

python create_tartu_raw.py --kitti_raw_root /gpfs/space/projects/ml2024 --target_dir /gpfs/space/projects/ml2024/Bolt/liso/tartu_dataset --pcd_locations /gpfs/space/projects/Bolt/liso/tartu_pcd_locations
