###### FOR TRAINING
# CUDA_VISIBLE_DEVICES='4,5,6,7' python -m torch.distributed.launch --master_port 2 --nproc_per_node=4 \
# --use_env /root/repos/KPCA_code/Robust/main_train.py \
# --model deit_base_patch16_224 --batch-size 256 --data-path /root/tensorflow_datasets/downloads/manual/ \
# --output_dir /root/checkpoints/experiments/rpc_base/ \
# --wandb --API_Key f9b91afe90c0f06aa89d2a428bd46dac42640bff \
# --job_name base_baseline2 \
# --clip-grad 1.0 
# --robust --num_iter 4 --lambd 1 --layer 0 \
# --resume /root/checkpoints/experiments/rpc_small/31_07_2024_05:20:36_deit_small_patch16_224_checkpoint.pth

###### FOR ROBUSTNESS EVAL
# CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch --master_port 1 --nproc_per_node=4 --use_env eval_OOD.py \
# --model deit_tiny_patch16_224 --data-path /path/to/data/imagenet/ --output_dir /path/to/checkpoints/ \
# --robust --num_iter 4 --lambd 4 --layer 0 --resume /path/to/model/checkpoint/