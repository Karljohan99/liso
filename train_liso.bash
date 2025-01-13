#!/bin/bash

export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

ldd --version

# source $HOME/liso/install_extra_packages.bash

conda init && source $HOME/.bashrc && conda activate liso

python3 liso/kabsch/liso_cli.py --fast-test --summary-dir $HOME/liso/data/train_liso/first_train -c nuscenes bev_100m_512 centerpoint batch_size_four liso
# python3 liso/kabsch/liso_cli.py --summary-dir YOUR_DESIRED_LOG_DIR_HERE -c waymo bev_100m_512 centerpoint batch_size_four liso gt_odom -kv data flow_source gt
