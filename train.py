
import os
import sys
import time
import json
import wandb
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, sampler
from tqdm import tqdm
import matplotlib.pyplot as plt

from scipy.interpolate import UnivariateSpline

import numpy as np
import cv2
import math

from argument import get_args
from backbone import resnet18, resnet50, resnet101, darknet_tiny, darknet53
from dataset import BOP_Dataset, collate_fn
from model import PoseModule

import transform
from evaluate import (
    evaluate_pose_predictions, 
    remap_predictions
)
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    DistributedSampler,
    all_gather,
)
from utils import (
    load_bop_meshes,
    load_bbox_3d,
    visualize_pred,
    print_accuracy_per_class,
    network_grad_ratio,
    json_dumps_numpy,
)
from tensorboardX import SummaryWriter

# reproducibility: https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(0)
np.random.seed(0)

# close shared memory of pytorch
if True:
    # https://github.com/huaweicloud/dls-example/issues/26
    from torch.utils.data import dataloader
    from torch.multiprocessing import reductions
    from multiprocessing.reduction import ForkingPickler
    default_collate_func = dataloader.default_collate
    def default_collate_override(batch):
        dataloader._use_shared_memory = False
        return default_collate_func(batch)
    setattr(dataloader, 'default_collate', default_collate_override)
    for t in torch._storage_classes:
        if sys.version_info[0] == 2:
            if t in ForkingPickler.dispatch:
                del ForkingPickler.dispatch[t]
        else:
            if t in ForkingPickler._extra_reducers:
                del ForkingPickler._extra_reducers[t]

def accumulate_dicts(data):
    all_data = all_gather(data)

    if get_rank() != 0:
        return

    data = {}

    for d in all_data:
        data.update(d)

    return data

@torch.no_grad()
def valid(cfg, steps, loader, model, device, logger=None):
    torch.cuda.empty_cache()

    model.eval()

    if get_rank() == 0:
        pbar = tqdm(enumerate(loader), total=len(loader), dynamic_ncols=True)
    else:
        pbar = enumerate(loader)

    preds = {}

    meshes, _ = load_bop_meshes(cfg['DATASETS']['MESH_DIR'])
    bboxes_3d = load_bbox_3d(cfg['DATASETS']['BBOX_FILE'])

    for bIdx, (images, targets, meta_infos) in pbar:
        model.zero_grad()

        images = images.to(device)
        targets = [target.to(device) for target in targets]
        pred, aux = model(images, targets=targets)

        # pred = [p.to('cpu') for p in pred]
        iIdx = 0
        for m, p in zip(meta_infos, pred):
            iIdx += 1
            new_p = remap_predictions(
                cfg['INPUT']['INTERNAL_K'], 
                cfg['INPUT']['INTERNAL_WIDTH'], 
                cfg['INPUT']['INTERNAL_HEIGHT'], 
                bboxes_3d, m, p
                )
            # 
            if get_rank() == 0 and iIdx == 1:  # only save the first on in the batch
            # if get_rank() == 0:
                imgpath, imgname = os.path.split(m['path'])
                name_prefix = imgpath.replace(os.sep, '_').replace('.', '') + '_' + os.path.splitext(imgname)[0]
                # name_prefix = (("%03d-%05d") % (epoch+1, idx))
                # if True:
                if False:
                    # save the ground truth pose and prediction pose
                    np.savetxt(cfg['RUNTIME']['WORKING_DIR'] + name_prefix + '_gt_R.txt', targets[bIdx].rotations[0].to('cpu').numpy())
                    np.savetxt(cfg['RUNTIME']['WORKING_DIR'] + name_prefix + '_gt_T.txt', targets[bIdx].translations[0].to('cpu').numpy())
                    np.savetxt(cfg['RUNTIME']['WORKING_DIR'] + name_prefix + '_pred_R.txt', pred[bIdx][0][2])
                    np.savetxt(cfg['RUNTIME']['WORKING_DIR'] + name_prefix + '_pred_T.txt', pred[bIdx][0][3])
                    
                if True:
                # if False:
                    visImg = visualize_pred(m, new_p, cfg['DATASETS']['SYMMETRY_TYPES'], meshes, bboxes_3d, cfg['DATASETS']['MESH_DIAMETERS'])
                    cv2.imwrite(cfg['RUNTIME']['WORKING_DIR'] + name_prefix + '_vis.jpg', visImg)
                    # 
                    # cv2.imshow("pred", visImg)
                    # cv2.waitKey(0)
            #
            preds.update({m['path']:{
                'meta': m,
                'pred': new_p
            }})
        #

    preds = accumulate_dicts(preds)

    if get_rank() != 0:
        return

    # write predictions to file
    json_file_name = cfg['RUNTIME']['WORKING_DIR'] + "preds.json"
    preds_formatted = json_dumps_numpy(preds)
    with open(json_file_name, 'w') as f:
        f.write(preds_formatted)
    # reload
    with open(json_file_name, 'r') as f:
        preds = json.load(f)
        
    accuracy_adi_per_class, accuracy_auc_per_class, accuracy_rep_per_class, accuracy_adi_per_depth, accuracy_rep_per_depth, depth_range \
        = evaluate_pose_predictions(preds, cfg['DATASETS']['N_CLASS'], meshes, cfg['DATASETS']['MESH_DIAMETERS'], cfg['DATASETS']['SYMMETRY_TYPES'])

    print_accuracy_per_class(accuracy_adi_per_class, accuracy_auc_per_class, accuracy_rep_per_class)
    # print(accuracy_adi_per_depth)

    # writing log to tensorboard
    if logger:
        classNum = cfg['DATASETS']['N_CLASS'] - 1 # get rid of background class        
        assert(len(accuracy_adi_per_class) == classNum)
        assert(len(accuracy_rep_per_class) == classNum)

        all_adi = {}
        all_rep = {}
        validClassNum = 0

        for i in range(classNum):
            className = ('class_%02d' % i)
            logger.add_scalars('ADI/' + className, accuracy_adi_per_class[i], steps)
            logger.add_scalars('REP/' + className, accuracy_rep_per_class[i], steps)
            wandb.log({'ADI/' + className: accuracy_adi_per_class[i],
                       'REP/' + className: accuracy_rep_per_class[i]},
                      step=steps)
            #
            assert(len(accuracy_adi_per_class[i]) == len(accuracy_rep_per_class[i]))
            if len(accuracy_adi_per_class[i]) > 0:
                for key, val in accuracy_adi_per_class[i].items():
                    if key in all_adi:
                        all_adi[key] += val
                    else:
                        all_adi[key] = val
                for key, val in accuracy_rep_per_class[i].items():
                    if key in all_rep:
                        all_rep[key] += val
                    else:
                        all_rep[key] = val
                validClassNum += 1

        # averaging
        for key, val in all_adi.items():
            all_adi[key] = val / validClassNum
        for key, val in all_rep.items():
            all_rep[key] = val / validClassNum  
        logger.add_scalars('ADI/all_class', all_adi, steps)
        logger.add_scalars('REP/all_class', all_rep, steps)
        wandb.log({'ADI/all_class': all_adi,
                   'REP/all_class': all_rep},
                  step=steps)

        #
        # assert(len(accuracy_adi_per_depth) == len(accuracy_rep_per_depth))
        # depth_bins = len(accuracy_adi_per_depth)
        # for i in range(depth_bins):
        #     depthName = ('depth_%01d' % i)
        #     logger.add_scalars('ADI/' + depthName, accuracy_adi_per_depth[i], steps)
        #     logger.add_scalars('REP/' + depthName, accuracy_rep_per_depth[i], steps)

    return accuracy_adi_per_class, accuracy_rep_per_class, accuracy_adi_per_depth, accuracy_rep_per_depth, depth_range

def gradient_trace_debug(cfg, steps, loader, model, optimizer, device, logger):
    loss_weight_combos = {
        "CLS_only": [cfg['SOLVER']['LOSS_WEIGHT_CLS'], 0], 
        "REG_only": [0, cfg['SOLVER']['LOSS_WEIGHT_REG']],
        #"CLS_and_REG": [cfg['SOLVER']['LOSS_WEIGHT_CLS'], cfg['SOLVER']['LOSS_WEIGHT_REG']]
        }
    
    for combo_key in loss_weight_combos:
        # # save model and optimizer first
        # tmp_pth_file = cfg['RUNTIME']['WORKING_DIR'] + 'gradient_trace_debug.pth'
        # torch.save({
        #     'model': model.state_dict(),
        #     'optim': optimizer.state_dict(),
        #     },
        #     tmp_pth_file,
        # )
        
        model.train()
        maxgrads = []
        avggrads = []
        avgdatas = []
        for idx, (images, targets, _) in enumerate(loader):
            model.zero_grad()
            
            images = images.to(device)
            targets = [target.to(device) for target in targets]
            
            _, loss_dict = model(images, targets=targets)
            loss_dict['loss_cls'] = loss_dict['loss_cls'] * loss_weight_combos[combo_key][0]
            loss_dict['loss_reg'] = loss_dict['loss_reg'] * loss_weight_combos[combo_key][1]
            loss_cls = loss_dict['loss_cls'].mean()
            loss_reg = loss_dict['loss_reg'].mean()
                
            loss = loss_cls + loss_reg   
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg['SOLVER']['GRAD_CLIP'])

            # do not actually update the model, only for gradient debug here
            # optimizer.step() 
            
            # neither touch the scheduler
            # scheduler.step()
            
            # loss_reduced = reduce_loss_dict(loss_dict)
            # loss_cls = loss_reduced['loss_cls'].mean().item()
            # loss_reg = loss_reduced['loss_reg'].mean().item()
            
            if get_rank() == 0:
                # gradient debug
                maxgrad, avggrad, avgdata = network_grad_ratio(model)
                # print('gradiant debug: g/<%.4f, %.4f>, d/%.4f, r/%.4f' % (maxgrad, avggrad, avgdata, avggrad / avgdata))
                maxgrads.append(maxgrad)
                avggrads.append(avggrad)
                avgdatas.append(avgdata)
                
            if idx >= 100: # 100 times is enough
                break;
            
        # writing log to tensorboard 
        logger.add_scalars('training/grad_abs_data', {combo_key: np.array(avgdatas).mean()}, steps)
        logger.add_scalars('training/grad_avg', {combo_key: np.array(avggrads).mean()}, steps)
        logger.add_scalars('training/grad_max', {combo_key: np.array(maxgrads).mean()}, steps)
        
        print('%s gradiant debug: abs_data/%.4f, avg_grad/%.4f, max_grad/%.4f' % (
            combo_key, np.array(avgdatas).mean(), 
            np.array(avggrads).mean(), 
            np.array(maxgrads).mean()
            ))
        
        # # recover model and optimizer
        # chkpt = torch.load(tmp_pth_file, map_location='cpu') 
        # model.load_state_dict(chkpt['model']) 
        # optimizer.load_state_dict(chkpt['optim'])

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return sampler.RandomSampler(dataset)

    else:
        return sampler.SequentialSampler(dataset)


if __name__ == '__main__':
    cfg = get_args()

    wandb.login(key="4893a49e39ba791d5d4c6c1eec868f2c99da0172")

    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    cfg['RUNTIME']['N_GPU'] = n_gpu
    cfg['RUNTIME']['DISTRIBUTED'] = n_gpu > 1

    if cfg['RUNTIME']['DISTRIBUTED']:
        torch.cuda.set_device(cfg['RUNTIME']['LOCAL_RANK'])
        torch.distributed.init_process_group(backend='gloo', init_method='env://')
        synchronize()

    # device = 'cuda'
    device = cfg['RUNTIME']['RUNNING_DEVICE']

    internal_K = np.array(cfg['INPUT']['INTERNAL_K']).reshape(3,3)

    train_trans = transform.Compose(
        [
            transform.Resize(
                cfg['INPUT']['INTERNAL_WIDTH'], 
                cfg['INPUT']['INTERNAL_HEIGHT'], internal_K),
            transform.RandomOcclusion(cfg['SOLVER']['AUGMENTATION_OCCLUSION']),
            transform.RandomBackground(cfg['SOLVER']['AUGMENTATION_BACKGROUND_DIR']),
            transform.RandomShiftScaleRotate(
                cfg['SOLVER']['AUGMENTATION_SHIFT'], 
                cfg['SOLVER']['AUGMENTATION_SCALE'], 
                cfg['SOLVER']['AUGMENTATION_ROTATION'], 
                cfg['INPUT']['INTERNAL_WIDTH'], 
                cfg['INPUT']['INTERNAL_HEIGHT'], 
                internal_K),
            transform.RandomHSV(
                cfg['SOLVER']['AUGMENTATION_ColorH'], 
                cfg['SOLVER']['AUGMENTATION_ColorS'], 
                cfg['SOLVER']['AUGMENTATION_ColorV']
                ),
            transform.RandomPencilSharpen(cfg['SOLVER']['AUGMENTATION_Sharpen']),
            transform.RandomSmooth(cfg['SOLVER']['AUGMENTATION_Smooth']),
            transform.RandomNoise(cfg['SOLVER']['AUGMENTATION_Noise']),
            transform.Grayscalize(cfg['SOLVER']['AUGMENTATION_Grayscalize']),
            transform.Normalize(
                cfg['INPUT']['PIXEL_MEAN'], 
                cfg['INPUT']['PIXEL_STD']),
            transform.ToTensor(),
        ]
    )

    valid_trans = transform.Compose(
        [
            transform.Resize(
                cfg['INPUT']['INTERNAL_WIDTH'], 
                cfg['INPUT']['INTERNAL_HEIGHT'], 
                internal_K),
            transform.Grayscalize(cfg['SOLVER']['AUGMENTATION_Grayscalize']),
            transform.Normalize(
                cfg['INPUT']['PIXEL_MEAN'], 
                cfg['INPUT']['PIXEL_STD']),
            transform.ToTensor(), 
        ]
    )

    train_set = BOP_Dataset(
        cfg['DATASETS']['TRAIN'], 
        cfg['DATASETS']['MESH_DIR'], 
        cfg['DATASETS']['BBOX_FILE'], 
        train_trans,
        cfg['DATASETS']['SYMMETRY_TYPES'],
        training = True)
    valid_set = BOP_Dataset(
        cfg['DATASETS']['VALID'],
        cfg['DATASETS']['MESH_DIR'], 
        cfg['DATASETS']['BBOX_FILE'], 
        valid_trans,
        training = False)

    if cfg['MODEL']['BACKBONE'] == 'resnet18':
        backbone = resnet18(pretrained=True)
    elif cfg['MODEL']['BACKBONE'] == 'resnet50':
        backbone = resnet50(pretrained=True)
    elif cfg['MODEL']['BACKBONE'] == 'resnet101':
        backbone = resnet101(pretrained=True)
    elif cfg['MODEL']['BACKBONE'] == 'darknet_tiny':
        backbone = darknet_tiny(pretrained=True)
    elif cfg['MODEL']['BACKBONE'] == 'darknet53':
        backbone = darknet53(pretrained=True)

    # wandb.tensorboard.patch(root_logdir=[cfg['RUNTIME']['WORKING_DIR'],
    #                                      cfg['RUNTIME']['WORKING_DIR'] + "ADI/all_class",
    #                                      cfg['RUNTIME']['WORKING_DIR'] + "REP/all_class"])
    wandb.init(project="single-pose-plus", entity="garrofederico",sync_tensorboard=True, config=cfg)
    # access all HPs through wandb.config, so logging matches execution!
    #cfg = wandb.config
    model = PoseModule(cfg, backbone)
    model = model.to(device)

    # https://discuss.pytorch.org/t/is-average-the-correct-way-for-the-gradient-in-distributeddataparallel-with-multi-nodes/34260/13
    base_lr = cfg['SOLVER']['BASE_LR'] / cfg['RUNTIME']['N_GPU']
    
    #optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001, nesterov=True)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.0001, eps=1e-8)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, base_lr, cfg['SOLVER']['MAX_ITER']+100, pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    batch_size_per_gpu = int(cfg['SOLVER']['IMS_PER_BATCH'] / cfg['RUNTIME']['N_GPU'])

    if cfg['RUNTIME']['DISTRIBUTED']:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[cfg['RUNTIME']['LOCAL_RANK']],
            output_device=cfg['RUNTIME']['LOCAL_RANK'],
            broadcast_buffers=False,
        )
        model = model.module

    if len(cfg['RUNTIME']['WORKING_DIR']) == 0:
        # create working_dir dynamically
        timestr = time.strftime('%Y%m%d_%H%M%S',time.localtime(time.time()))
        name_wo_ext = os.path.splitext(os.path.split(cfg['RUNTIME']['CONFIG_FILE'])[1])[0]
        cfg['RUNTIME']['WORKING_DIR'] = 'working_dirs' + '/' + name_wo_ext + '/' + timestr + '/'
    # 
    preload_file_name = None
    if os.path.exists(cfg['RUNTIME']['WORKING_DIR'] + 'latest.pth'):
        preload_file_name = cfg['RUNTIME']['WORKING_DIR'] + 'latest.pth'
    elif os.path.exists(cfg['RUNTIME']['WEIGHT_FILE']):
        preload_file_name = cfg['RUNTIME']['WEIGHT_FILE']

    total_steps = 0
    VAL_FREQ = cfg['SOLVER']['VAL_FREQ']

    if preload_file_name:
        try:
            chkpt = torch.load(preload_file_name, map_location='cpu')  # load checkpoint
            if 'model' in chkpt:
                assert('steps' in chkpt and 'optim' in chkpt)
                total_steps = chkpt['steps']
                model.load_state_dict(chkpt['model'])
                optimizer.load_state_dict(chkpt['optim'])
                scheduler.load_state_dict(chkpt['sched'])
                print('Weights, optimzer, scheduler are loaded from %s, starting from step %d' % (preload_file_name, total_steps))
            else:
                model.load_state_dict(chkpt)
                print('Weights from are loaded from ' + preload_file_name)
        except:
            pass
    else:
        pass
    # 
    print("working directory: " + cfg['RUNTIME']['WORKING_DIR'])
    if get_rank() == 0:
        os.makedirs(cfg['RUNTIME']['WORKING_DIR'], exist_ok=True)
        logger = SummaryWriter(cfg['RUNTIME']['WORKING_DIR'])

    # compute model size
    total_params_count = sum(p.numel() for p in model.parameters())
    print("Model size: %d parameters" % total_params_count)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size_per_gpu,
        sampler=data_sampler(train_set, shuffle=True, distributed=cfg['RUNTIME']['DISTRIBUTED']),
        num_workers=cfg['RUNTIME']['NUM_WORKERS'],
        collate_fn=collate_fn(cfg['INPUT']['SIZE_DIVISIBLE']),
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=batch_size_per_gpu,
        sampler=data_sampler(valid_set, shuffle=False, distributed=cfg['RUNTIME']['DISTRIBUTED']),
        num_workers=cfg['RUNTIME']['NUM_WORKERS'],
        collate_fn=collate_fn(cfg['INPUT']['SIZE_DIVISIBLE']),
    )

    # write cfg to working_dir
    with open(cfg['RUNTIME']['WORKING_DIR'] + 'cfg.json', 'w') as f:
        json.dump(cfg, f, indent=4, sort_keys=True)
        
    # gradient_trace_debug(cfg, total_steps, train_loader, model, optimizer, device, logger)
    # valid(cfg, total_steps, valid_loader, model, device, logger=logger)
    model.train()
    #wandb.watch(model, log="all", log_freq=10)

    should_keep_training = True
    while should_keep_training:
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), dynamic_ncols=True)
        for idx, (images, targets, _) in pbar:
            if total_steps >= cfg['SOLVER']['MAX_ITER']:
                should_keep_training = False
                torch.save(model.state_dict(), cfg['RUNTIME']['WORKING_DIR'] + 'final.pth')
                print('Training finished')
                wandb.save(cfg['RUNTIME']['WORKING_DIR'] + 'final.pth')
                torch.onnx.export(model, images, cfg['RUNTIME']['WORKING_DIR'] + "model.onnx")
                wandb.save(cfg['RUNTIME']['WORKING_DIR'] + "model.onnx")
                break
            total_steps += 1
            # 
            model.zero_grad()

            images = images.to(device)
            targets = [target.to(device) for target in targets]

            _, loss_dict = model(images, targets=targets)
            loss_dict['loss_cls'] = loss_dict['loss_cls'] * cfg['SOLVER']['LOSS_WEIGHT_CLS']
            loss_dict['loss_reg'] = loss_dict['loss_reg'] * cfg['SOLVER']['LOSS_WEIGHT_REG']
            loss_cls = loss_dict['loss_cls'].mean()
            loss_reg = loss_dict['loss_reg'].mean()

            loss = loss_cls + loss_reg
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg['SOLVER']['GRAD_CLIP'])
            optimizer.step()
            scheduler.step()

            # loss_reduced = reduce_loss_dict(loss_dict)
            # loss_cls = loss_reduced['loss_cls'].mean().item()
            # loss_reg = loss_reduced['loss_reg'].mean().item()

            if get_rank() == 0:
                current_lr = optimizer.param_groups[0]['lr']
                pbar_str = (("steps: %d/%d, lr:%.6f, cls:%.4f, reg:%.4f") % (total_steps, cfg['SOLVER']['MAX_ITER'], current_lr, loss_cls, loss_reg))
                pbar.set_description(pbar_str)

                # gradient debug
                # maxgrad, avggrad, avgdata = network_grad_ratio(model)
                # print('gradiant debug: g/<%.4f, %.4f>, d/%.4f, r/%.4f' % (maxgrad, avggrad, avgdata, avggrad / avgdata))

                # writing log to tensorboard
                if logger and idx % 10 == 0:
                    logger.add_scalar('training/learning_rate', current_lr, total_steps)
                    logger.add_scalar('training/loss_cls', loss_cls, total_steps)
                    logger.add_scalar('training/loss_reg', loss_reg, total_steps)
                    logger.add_scalar('training/loss_all', (loss_cls + loss_reg), total_steps)
                    wandb.log({"training/learning_rate": current_lr,
                               "training/loss_cls": loss_cls,
                               "training/loss_reg": loss_reg,
                               "training/loss_all": (loss_cls + loss_reg)
                               }, step=total_steps)
                if total_steps % VAL_FREQ == 0:
                    valid(cfg, total_steps, valid_loader, model, device, logger=logger)
                    model.train()
                    torch.save({
                        'steps': total_steps,
                        'model': model.state_dict(), 
                        'optim': optimizer.state_dict(),
                        'sched': scheduler.state_dict(),
                        },
                        cfg['RUNTIME']['WORKING_DIR'] + 'latest.pth',
                    )
                    wandb.save(cfg['RUNTIME']['WORKING_DIR'] + 'latest.pth')

    # output final info
    if get_rank() == 0:
        timestr = time.strftime('%Y%m%d_%H%M%S',time.localtime(time.time()))
        commandstr = ' '.join([str(elem) for elem in sys.argv]) 
        final_msg = ("finished at: %s\nworking_dir: %s\ncommands:%s" % (timestr, cfg['RUNTIME']['WORKING_DIR'], commandstr))
        with open(cfg['RUNTIME']['WORKING_DIR'] + 'info.txt', 'w') as f:
            f.write(final_msg)
        print(final_msg)
