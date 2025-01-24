#!/bin/bash

conda init && source $HOME/.bashrc && conda activate liso

python3 liso/slim/cli.py --inference-only --summary-dir /mnt/Bolt/liso/slim_infere_logs --load_checkpoint /mnt/Bolt/liso/train_logs/364e6/20250123_100410/checkpoints/2000.pth


