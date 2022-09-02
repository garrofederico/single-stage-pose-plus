import os

import numpy as np

import torch
from torch import nn
from boxlist import cat_boxlist, boxlist_iou

import math

INF = 100000000

class SigmoidFocalLoss(nn.Module):
    def __init__(self, gamma, alpha):
        super().__init__()

        self.gamma = gamma
        self.alpha = alpha
        self.eps = 1e-4

    def forward(self, out, target):
        n_class = out.shape[1]
        class_ids = torch.arange(
            1, n_class + 1, dtype=target.dtype, device=target.device
        ).unsqueeze(0)

        t = target.unsqueeze(1)
        #p = torch.sigmoid(out)
        p = torch.clamp(torch.sigmoid(out), min=self.eps, max=1-self.eps)

        gamma = self.gamma
        alpha = self.alpha

        term1 = (1 - p) ** gamma * torch.log(p)
        term2 = p ** gamma * torch.log(1 - p)

        # print(term1.sum(), term2.sum())

        loss = (
            -(t == class_ids).float() * alpha * term1
            - ((t != class_ids) * (t >= 0)).float() * (1 - alpha) * term2
        )

        return loss.sum()

def get_num_gpus():
    return int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

def reduce_sum(tensor):
    if get_num_gpus() <= 1:
        return tensor
    import torch.distributed as dist
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.reduce_op.SUM)
    return tensor

def cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)

def permute_and_flatten(layer, N, A, C, H, W):
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer

def concat_box_prediction_layers(pred_cls, pred_reg):
    pred_cls_flattened = []
    pred_reg_flattened = []
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the pred_reg
    for pred_cls_per_level, pred_reg_per_level in zip(
        pred_cls, pred_reg
    ):
        N, AxC, H, W = pred_cls_per_level.shape
        Cx16 = pred_reg_per_level.shape[1]
        C = Cx16 // 16
        A = 1
        pred_cls_per_level = permute_and_flatten(
            pred_cls_per_level, N, A, C, H, W
        )
        pred_cls_flattened.append(pred_cls_per_level)

        pred_reg_per_level = permute_and_flatten(
            pred_reg_per_level, N, A, (C*16), H, W
        )
        pred_reg_flattened.append(pred_reg_per_level)
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    pred_cls = cat(pred_cls_flattened, dim=1).reshape(-1, C)
    pred_reg = cat(pred_reg_flattened, dim=1).reshape(-1, C*16)
    return pred_cls, pred_reg

class PoseLoss(object):
    def __init__(self, gamma, alpha, anchor_sizes, anchor_strides, positive_type, positive_num, positive_lambda,
                    top_k, internal_K, diameters, target_coder):
        self.cls_loss_func = SigmoidFocalLoss(gamma, alpha)
        # self.centerness_loss_func = nn.BCEWithLogitsLoss(reduction="sum")
        # self.matcher = Matcher(fg_iou_threshold, bg_iou_threshold, True)
        self.anchor_sizes = anchor_sizes
        self.anchor_strides = anchor_strides
        self.positive_type = positive_type
        self.positive_num = positive_num
        self.positive_lambda = positive_lambda
        self.top_k = top_k
        self.internal_K = internal_K
        self.target_coder = target_coder
        self.diameters = diameters

    def ImageSpaceLoss(self, pred, target_2D, cls_labels, anchors, weight=None):
        cellNum = pred.shape[0]
        pred_filtered = pred.view(cellNum, -1, 16)[torch.arange(cellNum), cls_labels]

        pred_xy = self.target_coder.decode(pred_filtered, anchors)
        target_xy = self.target_coder.decode(target_2D, anchors)

        scaling_factor = 2 # 2px
        losses = nn.SmoothL1Loss(reduction='none')(scaling_factor * pred_xy, scaling_factor * target_xy).view(cellNum, -1).mean(dim=1)
        losses = losses / scaling_factor
        #losses = nn.L1Loss(reduction='none')(pred_xy, target_xy).mean(dim=1)

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum()
        else:
            assert losses.numel() != 0
            return losses.sum()

    def ObjectSpaceLoss(self, pred, target_3D_in_camera_frame, cls_labels, anchors, weight=None):
        if not isinstance(self.diameters, torch.Tensor):
            self.diameters = torch.FloatTensor(self.diameters).to(device=pred.device).view(-1)
        
        diameter_ext = self.diameters[cls_labels.view(-1,1).repeat(1, 8*3).view(-1, 3, 1)]

        cellNum = pred.shape[0]
        pred_filtered = pred.view(cellNum, -1, 16)[torch.arange(cellNum), cls_labels]

        pred_xy = self.target_coder.decode(pred_filtered, anchors)
        # target_xy = self.target_coder.decode(target, anchors)

        pred_xy = pred_xy.view(-1,2,8).transpose(1,2).contiguous().view(-1,2)

        # construct normalized 2d
        B = torch.inverse(self.internal_K).mm(torch.cat((pred_xy.t(), torch.ones_like(pred_xy[:,0]).view(1,-1)), dim=0)).t()
        # compute projection matrices
        P = torch.bmm(B.view(-1, 3, 1), B.view(-1, 1, 3)) / torch.bmm(B.view(-1, 1, 3), B.view(-1, 3, 1))

        target_3D_in_camera_frame = target_3D_in_camera_frame.view(-1, 3, 1)
        px = torch.bmm(P, target_3D_in_camera_frame)

        target_3D_in_camera_frame = target_3D_in_camera_frame / diameter_ext
        px = px / diameter_ext
        # target_3D_in_camera_frame = target_3D_in_camera_frame / self.diameters.mean()
        # px = px / self.diameters.mean()
        
        scaling_factor = 50 # 0.02d
        losses = nn.SmoothL1Loss(reduction='none')(scaling_factor * px, scaling_factor * target_3D_in_camera_frame).view(cellNum, -1).mean(dim=1)
        losses = losses / scaling_factor
        #losses = nn.L1Loss(reduction='none')(px, target_3D_in_camera_frame).view(cellNum, -1).mean(dim=1)

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum()
        else:
            assert losses.numel() != 0
            return losses.sum()

    def prepare_targets(self, targets, anchors):
        cls_labels = []
        reg_targets = []
        aux_raw_boxes = []
        aux_3D_in_camera_frame = []
        level_cnt = len(anchors[0])
        for im_i in range(len(targets)):
            pose_targets_per_im = targets[im_i]
            bbox_targets_per_im = pose_targets_per_im.to_object_boxlist()
            assert bbox_targets_per_im.mode == "xyxy"
            bboxes_per_im = bbox_targets_per_im.bbox
            labels_per_im = pose_targets_per_im.class_ids + 1
            anchors_per_im = cat_boxlist(anchors[im_i])
            num_gt = bboxes_per_im.shape[0]
            assert(level_cnt == len(anchors[im_i]))
            # 
            rotations_per_im = pose_targets_per_im.rotations
            translations_per_im = pose_targets_per_im.translations
            mask_per_im = pose_targets_per_im.mask

            if self.positive_type == 'SSC':
                anchor_sizes_per_level_interest = self.anchor_sizes[:level_cnt]
                anchor_strides_per_level_interst = self.anchor_strides[:level_cnt]
                gt_object_sizes = bbox_targets_per_im.box_span()

                num_anchors_per_level = [len(anchors_per_level.bbox) for anchors_per_level in anchors[im_i]]
                
                anchors_cx_per_im = (anchors_per_im.bbox[:, 2] + anchors_per_im.bbox[:, 0]) / 2.0
                anchors_cy_per_im = (anchors_per_im.bbox[:, 3] + anchors_per_im.bbox[:, 1]) / 2.0
                anchors_cx_per_im = torch.clamp(anchors_cx_per_im, min = 0, max = mask_per_im.shape[1] - 1).long()
                anchors_cy_per_im = torch.clamp(anchors_cy_per_im, min = 0, max = mask_per_im.shape[0] - 1).long()

                mask_at_anchors = mask_per_im[anchors_cy_per_im, anchors_cx_per_im]
                mask_labels = []
                for gt_i in range(num_gt):
                    valid_mask = (mask_at_anchors == (gt_i+1))
                    mask_labels.append(valid_mask)
                mask_labels = torch.stack(mask_labels).t()
                mask_labels = mask_labels.long()

                # random selecting candidates from each level first
                candidate_idxs = [[] for i in range(num_gt)]
                start_idx = 0
                gt_sz = gt_object_sizes.view(1,-1).repeat(level_cnt,1)
                lv_sz = torch.FloatTensor(anchor_sizes_per_level_interest).type_as(gt_sz)
                lv_sz = lv_sz.view(-1,1).repeat(1,num_gt)
                dk = torch.log2(gt_sz/lv_sz).abs()
                nk = torch.exp(-self.positive_lambda * (dk * dk))
                nk = self.positive_num * nk / nk.sum(0, keepdim=True)
                nk = (nk + 0.5).int()
                for level in range(level_cnt):
                    end_idx = start_idx + num_anchors_per_level[level]
                    is_in_mask_per_level = mask_labels[start_idx:end_idx, :]
                    # 
                    for gt_i in range(num_gt):
                        posi_num = nk[level][gt_i]

                        valid_pos = is_in_mask_per_level[:, gt_i].nonzero().view(-1)
                        posi_num = min(posi_num, len(valid_pos))
                        # rand_idx = torch.randint(0, len(valid_pos), (int(posi_num),)) # randoms with replacement
                        rand_idx = torch.randperm(len(valid_pos))[:posi_num] # randoms without replacement
                        candi_pos = valid_pos[rand_idx] + start_idx
                        candidate_idxs[gt_i].append(candi_pos)
                    # 
                    start_idx = end_idx

                # flagging selected positions
                roi = torch.full_like(mask_labels, -INF)
                for gt_i in range(num_gt):
                    tmp_idx = torch.cat(candidate_idxs[gt_i], dim=0)
                    roi[tmp_idx, gt_i] = 1

                anchors_to_gt_values, anchors_to_gt_indexs = roi.max(dim=1)
                cls_labels_per_im = labels_per_im[anchors_to_gt_indexs]
                cls_labels_per_im[anchors_to_gt_values == -INF] = 0 # background setting

                mask_visibilities, _ = mask_labels.max(dim=1)
                # logical_and, introduced only after pytorch 1.5
                # ignored_indexs = torch.logical_and(mask_visibilities==1, cls_labels_per_im==0)
                ignored_indexs = (mask_visibilities == 1) * (cls_labels_per_im == 0)
                cls_labels_per_im[ignored_indexs] = -1 # positions within mask but not selected will not be touched

                # 
                matched_boxes = bboxes_per_im[anchors_to_gt_indexs]
                matched_classes = (labels_per_im - 1)[anchors_to_gt_indexs]
                matched_rotations = rotations_per_im[anchors_to_gt_indexs]
                matched_translations = translations_per_im[anchors_to_gt_indexs]
            elif self.positive_type == 'ATSS':
                num_anchors_per_level = [len(anchors_per_level.bbox) for anchors_per_level in anchors[im_i]]
                ious = boxlist_iou(anchors_per_im, bbox_targets_per_im)

                gt_cx = (bboxes_per_im[:, 2] + bboxes_per_im[:, 0]) / 2.0
                gt_cy = (bboxes_per_im[:, 3] + bboxes_per_im[:, 1]) / 2.0
                gt_points = torch.stack((gt_cx, gt_cy), dim=1)

                anchors_cx_per_im = (anchors_per_im.bbox[:, 2] + anchors_per_im.bbox[:, 0]) / 2.0
                anchors_cy_per_im = (anchors_per_im.bbox[:, 3] + anchors_per_im.bbox[:, 1]) / 2.0
                anchor_points = torch.stack((anchors_cx_per_im, anchors_cy_per_im), dim=1)

                distances = (anchor_points[:, None, :] - gt_points[None, :, :]).pow(2).sum(-1).sqrt()

                # Selecting candidates based on the center distance between anchor box and object
                candidate_idxs = []
                star_idx = 0
                for level, anchors_per_level in enumerate(anchors[im_i]):
                    end_idx = star_idx + num_anchors_per_level[level]
                    distances_per_level = distances[star_idx:end_idx, :]
                    topk = min(self.top_k, num_anchors_per_level[level])
                    _, topk_idxs_per_level = distances_per_level.topk(topk, dim=0, largest=False)
                    candidate_idxs.append(topk_idxs_per_level + star_idx)
                    star_idx = end_idx
                candidate_idxs = torch.cat(candidate_idxs, dim=0)

                # Using the sum of mean and standard deviation as the IoU threshold to select final positive samples
                candidate_ious = ious[candidate_idxs, torch.arange(num_gt)]
                iou_mean_per_gt = candidate_ious.mean(0)
                iou_std_per_gt = candidate_ious.std(0)
                iou_thresh_per_gt = iou_mean_per_gt + iou_std_per_gt
                is_pos = candidate_ious >= iou_thresh_per_gt[None, :]

                # Limiting the final positive samples’ center to object
                anchor_num = anchors_cx_per_im.shape[0]
                for ng in range(num_gt):
                    candidate_idxs[:, ng] += ng * anchor_num
                e_anchors_cx = anchors_cx_per_im.view(1, -1).expand(num_gt, anchor_num).contiguous().view(-1)
                e_anchors_cy = anchors_cy_per_im.view(1, -1).expand(num_gt, anchor_num).contiguous().view(-1)
                candidate_idxs = candidate_idxs.view(-1)
                l = e_anchors_cx[candidate_idxs].view(-1, num_gt) - bboxes_per_im[:, 0]
                t = e_anchors_cy[candidate_idxs].view(-1, num_gt) - bboxes_per_im[:, 1]
                r = bboxes_per_im[:, 2] - e_anchors_cx[candidate_idxs].view(-1, num_gt)
                b = bboxes_per_im[:, 3] - e_anchors_cy[candidate_idxs].view(-1, num_gt)
                is_in_gts = torch.stack([l, t, r, b], dim=1).min(dim=1)[0] > 0.01
                is_pos = is_pos & is_in_gts

                # if an anchor box is assigned to multiple gts, the one with the highest IoU will be selected.
                ious_inf = torch.full_like(ious, -INF).t().contiguous().view(-1)
                index = candidate_idxs.view(-1)[is_pos.view(-1)]
                ious_inf[index] = ious.t().contiguous().view(-1)[index]
                ious_inf = ious_inf.view(num_gt, -1).t()

                anchors_to_gt_values, anchors_to_gt_indexs = ious_inf.max(dim=1)
                cls_labels_per_im = labels_per_im[anchors_to_gt_indexs]
                cls_labels_per_im[anchors_to_gt_values == -INF] = 0
                # 
                matched_boxes = bboxes_per_im[anchors_to_gt_indexs]
                matched_classes = (labels_per_im - 1)[anchors_to_gt_indexs]
                matched_rotations = rotations_per_im[anchors_to_gt_indexs]
                matched_translations = translations_per_im[anchors_to_gt_indexs]
                matched_kp_positions = kp_postions_per_im[anchors_to_gt_indexs]
            elif self.positive_type == 'TOPK':
                gt_cx = (bboxes_per_im[:, 2] + bboxes_per_im[:, 0]) / 2.0
                gt_cy = (bboxes_per_im[:, 3] + bboxes_per_im[:, 1]) / 2.0
                gt_points = torch.stack((gt_cx, gt_cy), dim=1)

                anchors_cx_per_im = (anchors_per_im.bbox[:, 2] + anchors_per_im.bbox[:, 0]) / 2.0
                anchors_cy_per_im = (anchors_per_im.bbox[:, 3] + anchors_per_im.bbox[:, 1]) / 2.0
                anchor_points = torch.stack((anchors_cx_per_im, anchors_cy_per_im), dim=1)

                distances = (anchor_points[:, None, :] - gt_points[None, :, :]).pow(2).sum(-1).sqrt()
                distances = distances / distances.max() / 1000
                ious = boxlist_iou(anchors_per_im, targets_per_im)

                is_pos = ious * False
                for ng in range(num_gt):
                    _, topk_idxs = (ious[:, ng] - distances[:, ng]).topk(self.top_k, dim=0)
                    l = anchors_cx_per_im[topk_idxs] - bboxes_per_im[ng, 0]
                    t = anchors_cy_per_im[topk_idxs] - bboxes_per_im[ng, 1]
                    r = bboxes_per_im[ng, 2] - anchors_cx_per_im[topk_idxs]
                    b = bboxes_per_im[ng, 3] - anchors_cy_per_im[topk_idxs]
                    is_in_gt = torch.stack([l, t, r, b], dim=1).min(dim=1)[0] > 0.01
                    is_pos[topk_idxs[is_in_gt == 1], ng] = True

                ious[is_pos == 0] = -INF
                anchors_to_gt_values, anchors_to_gt_indexs = ious.max(dim=1)

                cls_labels_per_im = labels_per_im[anchors_to_gt_indexs]
                cls_labels_per_im[anchors_to_gt_values == -INF] = 0
                matched_gts = bboxes_per_im[anchors_to_gt_indexs]

            matched_3Ds = pose_targets_per_im.keypoints_3d[matched_classes]
            # matched_Ks = pose_targets_per_im.K.repeat(matched_classes.shape[0], 1, 1)
            # TODO
            # assert equals self K
            if not isinstance(self.internal_K, torch.Tensor):
                self.internal_K = torch.FloatTensor(self.internal_K).to(device=matched_3Ds.device).view(3, 3)
            reg_targets_per_im = self.target_coder.encode(
                self.internal_K, matched_3Ds, 
                matched_rotations, matched_translations,
                anchors_per_im.bbox)
            cls_labels.append(cls_labels_per_im)
            reg_targets.append(reg_targets_per_im)
            aux_raw_boxes.append(matched_boxes)
            matched_3D_in_camera_frame = torch.bmm(matched_rotations, matched_3Ds.transpose(1, 2)) + matched_translations
            aux_3D_in_camera_frame.append(matched_3D_in_camera_frame.transpose(1, 2))

        return cls_labels, reg_targets, aux_raw_boxes, aux_3D_in_camera_frame

    def __call__(self, pred_cls, pred_reg, targets, anchors):
        labels, reg_targets, aux_raw_boxes, aux_3D_in_camera_frame = self.prepare_targets(targets, anchors)

        N = len(labels)
        pred_cls_flatten, pred_reg_flatten = concat_box_prediction_layers(pred_cls, pred_reg)

        labels_flatten = torch.cat(labels, dim=0)
        reg_targets_flatten = torch.cat(reg_targets, dim=0)
        aux_raw_boxes_flatten = torch.cat(aux_raw_boxes, dim=0)
        aux_3D_in_camera_frame_flatten = torch.cat(aux_3D_in_camera_frame, dim=0)
        anchors_flatten = torch.cat([cat_boxlist(anchors_per_image).bbox for anchors_per_image in anchors], dim=0)

        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)
        
        num_gpus = get_num_gpus()
        total_num_pos = reduce_sum(pos_inds.new_tensor([pos_inds.numel()])).item()
        num_pos_avg_per_gpu = max(total_num_pos / float(num_gpus), 1.0)
        
        valid_cls_inds = torch.nonzero(labels_flatten >= 0).squeeze(1)
        cls_loss = self.cls_loss_func(pred_cls_flatten[valid_cls_inds], labels_flatten[valid_cls_inds]) 
        #/ num_pos_avg_per_gpu

        if pos_inds.numel() > 0:
            pred_reg_flatten = pred_reg_flatten[pos_inds]
            cls_label_flatten = labels_flatten[pos_inds] - 1 # start from class 0
            reg_targets_flatten = reg_targets_flatten[pos_inds]
            aux_raw_boxes_flatten = aux_raw_boxes_flatten[pos_inds]
            aux_3D_in_camera_frame_flatten = aux_3D_in_camera_frame_flatten[pos_inds]
            anchors_flatten = anchors_flatten[pos_inds]

            # sum_centerness_targets_avg_per_gpu = reduce_sum(centerness_targets.sum()).item() / float(num_gpus)
            # reg_loss = self.GIoULoss(pred_reg_flatten, reg_targets_flatten, anchors_flatten,
                                    #  weight=centerness_targets) / sum_centerness_targets_avg_per_gpu
            # reg_loss = self.ImageSpaceLoss(pred_reg_flatten, reg_targets_flatten, anchors_flatten, aux_3D_in_camera_frame_flatten,
                                    #  weight=centerness_targets) / sum_centerness_targets_avg_per_gpu
            if self.target_coder.target_type == '2D':
                reg_loss = self.ImageSpaceLoss(pred_reg_flatten, reg_targets_flatten, cls_label_flatten, anchors_flatten)
                #/ num_pos_avg_per_gpu
            elif self.target_coder.target_type == '3D':
                reg_loss = self.ObjectSpaceLoss(pred_reg_flatten, aux_3D_in_camera_frame_flatten, cls_label_flatten, anchors_flatten) 
                #/ num_pos_avg_per_gpu
            else:
                assert(0)
        else:
            reg_loss = pred_reg_flatten.sum()
            # centerness_loss = reg_loss * 0

        # return cls_loss, reg_loss, centerness_loss
        return cls_loss, reg_loss 
