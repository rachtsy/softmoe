source /root/rach/softmoe/.sftmoe/bin/activate

###### FOR ROBUSTNESS EVAL
CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch --master_port 1 --nproc_per_node=4 --use_env eval_OOD.py \
--model soft_moe_vit_tiny --data-path /root/ --output_dir /root/rach/softmoe/ckpts/ \
--resume /root/rach/softmoe/ckpts/baseline3.pth --use-wandb