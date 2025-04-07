source /root/rach/softmoe/.sftmoe/bin/activate

###### FOR ROBUSTNESS EVAL
CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch --master_port 12 --nproc_per_node=4 --use_env eval_OOD.py \
--model soft_graph_moe_vit_tiny --data-path /root/ --output_dir /root/rach/softmoe/ckpts/ \
--resume /root/rach/softmoe/ckpts/graph_0.9_0.9_perepoch_DD_t-0.5.pth --t 0.5 --use-wandb