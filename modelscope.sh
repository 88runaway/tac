modelscope download \
  --model JianboZhou/pi_all_128_20k \
  --local_dir /data1/zjb/ckpt/lerobot/pi05_jax/all/128_20k




modelscope download \
  --dataset byml2024/UniVTAC \
  --include "insert_HDMI/clean/*" \
  --local_dir /data1/zjb/ckpt/UniVTAC