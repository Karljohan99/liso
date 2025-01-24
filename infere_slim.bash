#!/bin/bash

conda init && source $HOME/.bashrc && conda activate liso

python3 liso/slim/cli.py --inference-only --summary-dir /mnt/Bolt/liso/kitti_slim_infere_logs --load_checkpoint /mnt/Bolt/liso/kitti_train_logs/f39a0/20250123_150226/checkpoints/15000.pth


