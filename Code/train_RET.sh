torchrun --nnodes=1 --nproc_per_node=2 scripts/train.py \
--save_dir experiments \
--exp_name smplx_S2G \
--speakers oliver seth conan chemistry \
--config_file ./config/EDIT.json