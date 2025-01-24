#!/bin/bash

conda init && source $HOME/.bashrc && conda activate liso

python3 liso/slim/cli.py --inference-only --summary-dir /mnt/Bolt/liso/kitti_slim_infere_logs --load_checkpoint /mnt/Bolt/liso/kitti_train_logs/4b3c3/20250124_103751/checkpoints/500.pth


