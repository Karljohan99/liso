#!/bin/bash

export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

ldd --version

# source $HOME/liso/install_extra_packages.bash

conda init && source $HOME/.bashrc && conda activate liso

python3 liso/slim/cli.py --inference-only --summary-dir $HOME/liso/data/train_slim/first_inference --load_checkpoint $HOME/liso/data/train_slim/first_train/b177b/20250110_155802/checkpoints/15000.pth


