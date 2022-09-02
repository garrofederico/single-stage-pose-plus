import argparse
import yaml
import time
import os

def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--config_file', type=str, default='./configs/vespa.yaml')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--working_dir', type=str, default='./outputs/')
    parser.add_argument('--test_file', type=str, default='')
    parser.add_argument('--weight_file', type=str, default='')
    # parser.add_argument('--weight_file', type=str, default='./pretrained/vespa_pbr.pth')
    # parser.add_argument('--weight_file', type=str, default='./outputs_ptfall-pbr-hsv/final.pth')
    # parser.add_argument('--weight_file', type=str, default='./final_swisscube.pth')
    parser.add_argument('--running_device', type=str, default='cuda')
    

    return parser

def get_args():
    parser = get_argparser()
    args = parser.parse_args()

    # Read yaml configure
    with open(args.config_file, 'r') as load_f:
        cfg = yaml.load(load_f, Loader=yaml.FullLoader)

    cfg['RUNTIME'] = {}
    cfg['RUNTIME']['LOCAL_RANK'] = args.local_rank
    cfg['RUNTIME']['CONFIG_FILE'] = args.config_file
    cfg['RUNTIME']['NUM_WORKERS'] = args.num_workers
    cfg['RUNTIME']['WEIGHT_FILE'] = args.weight_file
    cfg['RUNTIME']['WORKING_DIR'] = args.working_dir
    cfg['RUNTIME']['RUNNING_DEVICE'] = args.running_device
    #
    if len(args.test_file) > 0:
        cfg['DATASETS']['TEST'] = args.test_file
    #
    if cfg['MODEL']['BACKBONE'] == 'resnet18':
        cfg['MODEL']['FEAT_CHANNELS'] = [0, 0, 128, 256, 512]
    elif cfg['MODEL']['BACKBONE'] in ['resnet50', 'resnet101']:
        cfg['MODEL']['FEAT_CHANNELS'] = [0, 0, 512, 1024, 2048]
    elif cfg['MODEL']['BACKBONE'] == 'darknet_tiny':
        cfg['MODEL']['FEAT_CHANNELS'] = [0, 0, 128, 128]
    elif cfg['MODEL']['BACKBONE'] == 'darknet53':
        cfg['MODEL']['FEAT_CHANNELS'] = [0, 0, 256, 512, 1024]
    else:
        print('Unsupported backbone')
        assert(0)

    cfg['MODEL']['OUT_CHANNEL'] = 256
    # cfg['MODEL']['ANCHOR_STRIDES'] = [16]
    # cfg['MODEL']['OUT_CHANNEL'] = 128
    cfg['MODEL']['N_CONV'] = 4
    cfg['MODEL']['PRIOR'] = 0.01
    if 'USE_HIGHER_LEVELS' not in cfg['MODEL']:
        cfg['MODEL']['USE_HIGHER_LEVELS'] = True

    cfg['SOLVER']['FOCAL_GAMMA'] = 2.0
    cfg['SOLVER']['FOCAL_ALPHA'] = 0.25
    cfg['SOLVER']['TOP_K'] = 9
    cfg['SOLVER']['POSITIVE_NUM'] = 10

    cfg['INPUT']['PIXEL_MEAN'] = [0.485, 0.456, 0.406]
    cfg['INPUT']['PIXEL_STD'] = [0.229, 0.224, 0.225]
    cfg['INPUT']['SIZE_DIVISIBLE'] = 32

    if 'GRAD_CLIP' not in cfg['SOLVER']:
        cfg['SOLVER']['GRAD_CLIP'] = 1.0
    if 'VAL_FREQ' not in cfg['SOLVER']:
        cfg['SOLVER']['VAL_FREQ'] = 5000
    if 'AUGMENTATION_OCCLUSION' not in cfg['SOLVER']:
        cfg['SOLVER']['AUGMENTATION_OCCLUSION'] = 0
    if 'AUGMENTATION_Grayscalize' not in cfg['SOLVER']:
        cfg['SOLVER']['AUGMENTATION_Grayscalize'] = False
    if 'AUGMENTATION_Smooth' not in cfg['SOLVER']:
        cfg['SOLVER']['AUGMENTATION_Smooth'] = 0
    if 'AUGMENTATION_Sharpen' not in cfg['SOLVER']:
        cfg['SOLVER']['AUGMENTATION_Sharpen'] = 0
    if 'SYMMETRY_TYPES' not in cfg['DATASETS']:
        cfg['DATASETS']['SYMMETRY_TYPES'] = {} # nothing but a place holder
    if 'AUGMENTATION_BACKGROUND_DIR' not in cfg['SOLVER']:
        cfg['SOLVER']['AUGMENTATION_BACKGROUND_DIR'] = None

    return cfg
