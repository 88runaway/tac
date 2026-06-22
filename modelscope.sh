modelscope download \
  --model JianboZhou/pi_mot_128_18k \
  --local_dir /data1/zjb/ckpt/lerobot/pi05_jax/frozen_mot/18k




modelscope download \
  --dataset byml2024/UniVTAC \
  --include "insert_HDMI/clean/*" \
  --local_dir /data1/zjb/ckpt/UniVTAC