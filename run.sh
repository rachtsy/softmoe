###### FOR TRAINING
CUDA_VISIBLE_DEVICES='4,5,6,7' python -m torch.distributed.launch --master_port 10002 --nproc_per_node=4 --use_env main_train.py \
--model soft_moe_vit_tiny --batch-size 256 --data-path /root/tensorflow_datasets/downloads/manual/ --output_dir /root/checkpoints/softmoe/ \
--API_Key f9b91afe90c0f06aa89d2a428bd46dac42640bff 


