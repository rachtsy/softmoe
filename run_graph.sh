###### FOR TRAINING
source /root/rach/.sftmoe/bin/activate

CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch --master_port 10022 --nproc_per_node=4 --use_env main_train.py \
--model soft_graph_moe_vit_tiny --batch-size 256 --data-path /home/imagenet-1k --output_dir /root/rach/softmoe/ckpts \
--API_Key f9b91afe90c0f06aa89d2a428bd46dac42640bff --jobname graphtest


