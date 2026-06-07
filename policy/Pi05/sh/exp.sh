cd /data1/zjb/UniVTAC

# 2倍下采样 60 fps → 30fps，适配 pi0.5 预训练分布
python scripts/convert_to_lerobot.py --task lift_bottle --model tactile \
    --data_dir /data1/zjb/ckpt/UniVTAC/lift_bottle/clean \
    --output_dir /data1/zjb/UniVTAC/data_lerobot --overwrite

# 快速验证（只转10个episode，不写入）
python scripts/convert_to_lerobot.py \
    --task lift_can \
    --dry_run


# 可选： 使用Quantiles归一化（区别于mean_std归一化）
cd /data1/zjb/lerobot
python src/lerobot/scripts/augment_dataset_quantile_stats.py \
    --repo-id univtac/lift_bottle \
    --root /data1/zjb/UniVTAC/data_lerobot/insert_tube


# LORA微调
cd /data1/zjb/UniVTAC
bash policy/Pi05/sh/train.sh lift_bottle 2,3,4,5,6,7
bash policy/Pi05/sh/train.sh insert_tube 4,5 train_lora
bash policy/Pi05/sh/train.sh put_bottle_in_shelf 5,7 train_lora
bash policy/Pi05/sh/train.sh lift_bottle 0,3,4 train_lora --model=tactile