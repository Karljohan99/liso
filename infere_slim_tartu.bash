#!/bin/bash

conda init && source $HOME/.bashrc && conda activate liso

python3 liso/slim/cli.py --inference-only --summary-dir /mnt/Bolt/liso/slim_infere_logs --load_checkpoint /mnt/Bolt/liso/train_logs/70295/20250205_234012/checkpoints/13000.pth


