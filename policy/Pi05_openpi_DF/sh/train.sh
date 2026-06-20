cd /data1/zjb/UniVTAC
unset PYTHONPATH  # prevent IsaacLab env from polluting openpi numpy path
conda activate openpi

# 用已有的多任务微调 ckpt 做 warm start，继续 DF 训练
python policy/Pi05_openpi_DF/train_df.py \
    --task insert_HDMI \
    --gpu 4,5,6,7 \
    --fsdp_devices 4 \
    --warm_start_ckpt /data1/zjb/UniVTAC/ckpt/lerobot/pi05_jax/all/128_20k/params \
    --overwrite

python policy/Pi05_openpi_DF/train_df.py \
    --task insert_HDMI \
    --gpu 4,5,6,7 \
    --fsdp_devices 4 \
    --resume

# 1. 数据转换（带触觉）
python policy/Pi05_openpi_DF/convert_df_tactile.py --task insert_HDMI


# 2. 训练（启用触觉）
python policy/Pi05_openpi_DF/train_df.py --task insert_HDMI --gpu 3,4,5,6 \
    --use_tactile true --block_time_sampling monotone --mix_prob 1.0 \
    --warm_start_ckpt /data1/zjb/UniVTAC/ckpt/lerobot/pi05_jax/all/128_20k/params --overwrite

# 3. 评估（block 级触觉反馈）
# 在 deploy config 中设置 use_tactile: true