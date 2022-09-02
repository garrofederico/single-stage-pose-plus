#CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=1234 train.py --config_file ./configs/ycbv.yaml --num_workers 4
CUDA_VISIBLE_DEVICES=0 python3 train.py --config_file ./configs/vespa.yaml --num_workers 8
