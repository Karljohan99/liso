#!/bin/bash

conda init && source $HOME/.bashrc && conda activate liso

python3 liso/kabsch/liso_cli.py --summary-dir $/mnt/Bolt/liso/liso_train_logs -c tartu bev_100m_512 centerpoint batch_size_four liso
