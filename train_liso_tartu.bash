#!/bin/bash

export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

ldd --version

# source $HOME/liso/install_extra_packages.bash

conda init && source $HOME/.bashrc && conda activate liso

python3 liso/kabsch/liso_cli.py --summary-dir $HOME/liso/data/train_liso_tartu -c nuscenes bev_100m_512 centerpoint batch_size_four liso