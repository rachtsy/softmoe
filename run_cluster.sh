# ##### FOR INFERENCE
# ## CUDA_VISIBLE_DEVICES='0' python -m torch.distributed.launch --master_port 1 --nproc_per_node=1 --use_env main.py\
# ## --model deit_tiny_patch16_224 --batch-size 1000 --data-path path/to/data --output_dir some/random/path

###### FOR TRAINING
CUDA_VISIBLE_DEVICES='0' python -m torch.distributed.launch --master_port 10020 --nproc_per_node=1 --use_env main_cluster.py \
--model deit_tiny_patch16_224 --batch-size 256 --epochs 1 --data-path \
~/data/imagenet/ --output_dir ~/data/ --API_Key f9b91afe90c0f06aa89d2a428bd46dac42640bff --eval \
--resume /root/checkpoints/Asym/27_01_2024_02:04:37_deit_tiny_patch16_224_checkpoint.pth
# https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth



