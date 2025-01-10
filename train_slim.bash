#!/bin/bash

export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

ldd --version

source $HOME/liso/install_extra_packages.bash

conda init && source $HOME/.bashrc && conda activate liso

python3 liso/slim/cli.py --summary-dir $HOME/liso/data/train_slim/first_train -c slim_simple_knn_training slim_nuscenes use_lidar_intensity slim_RAFT batch_size_one slim_highest_resolution
