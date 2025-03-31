###### FOR TRAINING
source .sftmoe/bin/activate

CUDA_VISIBLE_DEVICES='4,5,6,7' python -m torch.distributed.launch --master_port 10022 --nproc_per_node=4 --use_env main_train.py \
--model soft_graph_moe_vit_tiny --batch-size 256 --data-path /root/Imagenet --output_dir /root/rach/softmoe/ckpts \
--API_Key f9b91afe90c0f06aa89d2a428bd46dac42640bff --jobname graph_0.9_0.9_perepoch_DD_t-0.5 --warmupover 0 --t 0.5


