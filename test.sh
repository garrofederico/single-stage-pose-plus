# CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=1234 test.py --config_file ./configs/vespa.yaml --num_workers 8
# CUDA_VISIBLE_DEVICES=0 python3 test.py --config_file ./configs/vespa.yaml --num_workers 8 --weight_file './data/darknet_tiny_vespa_0327.pth' --running_device 'cpu'
CUDA_VISIBLE_DEVICES=0 python3 test.py --config_file ./configs/speed.yaml --num_workers 8 --weight_file './data/speed_darknet_tiny_20200726.pth' --running_device 'cpu'
