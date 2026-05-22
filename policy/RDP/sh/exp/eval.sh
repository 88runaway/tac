# grasp_classify
CKPT_DIR=/data1/zjb/reactive_diffusion_policy/data/outputs/2026.05.21/19.02.47_train_latent_diffusion_unet_image_univtac_ldp_marker_emb \
PCA_DIR=/data1/zjb/reactive_diffusion_policy/data/PCA_Transform_UniVTAC_grasp_classify \
bash policy/RDP/sh/eval.sh grasp_classify univtac 1 true

# pull_out_key
CKPT_DIR=/data1/zjb/reactive_diffusion_policy/data/outputs/2026.05.21/19.48.39_train_ldp_marker_emb_pull_out_key \
PCA_DIR=/data1/zjb/reactive_diffusion_policy/data/PCA_Transform_UniVTAC_pull_out_key \
bash policy/RDP/sh/eval.sh pull_out_key univtac 2 true

# insert_hole
CKPT_DIR=/data1/zjb/reactive_diffusion_policy/data/outputs/2026.05.21/19.59.41_train_ldp_marker_emb_insert_hole \
PCA_DIR=/data1/zjb/reactive_diffusion_policy/data/PCA_Transform_UniVTAC_insert_hole \
bash policy/RDP/sh/eval.sh insert_hole univtac 3 true