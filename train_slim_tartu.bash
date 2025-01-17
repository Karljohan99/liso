#!/bin/bash

export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

conda init && source $HOME/.bashrc && conda activate liso

python3 liso/slim/cli.py --summary-dir /mnt/Bolt/train_logs -c slim_simple_knn_training slim_tartu use_lidar_intensity slim_RAFT batch_size_one slim_highest_resolution
