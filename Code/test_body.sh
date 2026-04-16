python -W ignore scripts/test_body.py \
--save_dir experiments \
--exp_name smplx_S2G \
--speakers oliver seth conan chemistry \
--config_file ./config/EDIT.json \
--body_model_name s2g_body_pixel \
--body_model_path ./experiments/2024-07-09-11clock-smplx_S2G-base-ViT/ckpt-99.pth \
--infer \
--testmode \
