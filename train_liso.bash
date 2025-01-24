#!/bin/bash

conda init && source $HOME/.bashrc && conda activate liso

python3 liso/kabsch/liso_cli.py --summary-dir /mnt/Bolt/liso/kitti_train_logs -c kitti bev_100m_512 centerpoint batch_size_four liso