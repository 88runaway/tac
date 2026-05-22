#!/bin/bash
# Collect random-touch data for PCA training.
#
# The robot opens its gripper and perturbs joint angles near a prism object
# so that diverse tactile contact deformations are recorded.  The resulting
# HDF5 files can be used to train a task-agnostic PCA model for the
# marker_emb tactile modality (see policy/RDP/sh/process_data_marker.sh).
#
# Usage:
#   bash sh/collect_random_touch.sh [PRISM_NAME] [GPU] [EPISODE_NUM] [CONFIG]
#
# Arguments:
#   PRISM_NAME   Shape of the contact object (default: Hemisphere)
#                Other options: Cylinder, Box, ...
#   GPU          CUDA device index (default: 0)
#   EPISODE_NUM  Number of successful episodes to collect (default: 50)
#   CONFIG       task_config name without .yml (default: random_touch)
#
# Examples:
#   bash sh/collect_random_touch.sh                        # hemisphere, gpu 0, 50 eps
#   bash sh/collect_random_touch.sh Cylinder 1 100         # cylinder, gpu 1, 100 eps
#   bash sh/collect_random_touch.sh Hemisphere 0 200 random_touch_dense

PRISM_NAME=${1:-Hemisphere}
GPU=${2:-0}
EPISODE_NUM=${3:-50}
CONFIG=${4:-random_touch}

echo "============================================"
echo " Random-touch data collection"
echo "  PRISM_NAME  : $PRISM_NAME"
echo "  GPU         : $GPU"
echo "  EPISODE_NUM : $EPISODE_NUM"
echo "  CONFIG      : $CONFIG"
echo "============================================"
echo "NOTE: Run this script in the UniVTAC conda environment:"
echo "  conda activate UniVTAC"
echo "  source IsaacLab/_isaac_sim/setup_conda_env.sh"
echo "============================================"

export PRISM_NAME="$PRISM_NAME"

python scripts/collect_contact.py \
    random_touch \
    "$CONFIG" \
    --gpu "$GPU"

echo "Done. Data saved under: data/random_touch/$PRISM_NAME/"
echo ""
echo "Next step — train PCA on the collected HDF5 data:"
echo "  conda activate rdp"
echo "  python /data1/zjb/reactive_diffusion_policy/scripts/generate_pca_univtac.py \\"
echo "      --hdf5_dir $ROOT_DIR/data/random_touch/$PRISM_NAME \\"
echo "      --output_dir /data1/zjb/reactive_diffusion_policy/data/PCA_Transform_RandomTouch \\"
echo "      --n_components 32"
