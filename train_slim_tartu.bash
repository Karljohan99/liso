#!/bin/bash

conda init && source $HOME/.bashrc && conda activate liso

python3 liso/slim/cli.py --summary-dir /mnt/Bolt/liso/train_logs -c slim_simple_knn_training slim_tartu use_lidar_intensity slim_RAFT batch_size_one slim_highest_resolution
