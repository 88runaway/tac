# grasp_classify
CKPT_DIR=/data1/zjb/reactive_diffusion_policy/data/outputs/2026.05.25/01.48.02_train_latent_diffusion_unet_image_univtac_ldp_marker_emb_dual_cam \
PCA_DIR=/data1/zjb/reactive_diffusion_policy/data/PCA_Transform_UniVTAC_lift_can \
bash policy/RDP/sh/eval.sh lift_can univtac 4 true

# pull_out_key
CKPT_DIR=/data1/zjb/reactive_diffusion_policy/data/outputs/2026.05.25/01.57.00_train_latent_diffusion_unet_image_univtac_ldp_marker_emb_dual_cam \
PCA_DIR=/data1/zjb/reactive_diffusion_policy/data/PCA_Transform_UniVTAC_insert_tube \
bash policy/RDP/sh/eval.sh insert_tube univtac 7 true

# insert_hole
CKPT_DIR=/data1/zjb/reactive_diffusion_policy/data/outputs/2026.05.21/19.59.41_train_ldp_marker_emb_insert_hole \
PCA_DIR=/data1/zjb/reactive_diffusion_policy/data/PCA_Transform_UniVTAC_insert_hole \
bash policy/RDP/sh/eval.sh insert_hole univtac 3 true