#!/bin/bash

conda init && source $HOME/.bashrc && conda activate liso

python3 liso/slim/cli.py --inference-only --summary-dir /mnt/Bolt/liso/slim_infere_logs --load_checkpoint /mnt/Bolt/liso/train_logs/d0cf1/20250125_185514/checkpoints/150000.pth


