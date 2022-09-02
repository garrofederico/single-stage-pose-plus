import torch
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt

import transforms3d

from boxlist import BoxList

from utils import (
    rotation2quaternion, 
    quaternion2rotation,
    generate_shiftscalerotate_matrix,
    get_single_bop_annotation,
    load_bop_meshes,
    draw_bounding_box,
    draw_pose_axis,
    remap_pose,
    pose_symmetry_handling,
    )

class PoseAnnot(object):
    """
    This class represents a set of 6D pose objects within one image
    """

    def __init__(self, bbox_3d, K, mask, class_ids, rotations, translations, width, height):
        self.keypoints_3d = bbox_3d
        self.K = K
        self.mask = mask
        self.class_ids = class_ids
        self.rotations = rotations
        self.translations = translations
        self.width = width
        self.height = height

    def transform(self, M, target_K, target_width, target_height):
        """
        M: the transform matrix
        target_K: the target intrinsic matrix
        """
        new_masks = cv2.warpAffine(self.mask, M[:2], (target_width, target_height), flags=cv2.INTER_NEAREST, borderValue=0)

        # compute new RT under internal K
        new_rotations = []
        new_translations = []
        for i in range(len(self.class_ids)):
            cls_id = self.class_ids[i]
            pt3d = np.array(self.keypoints_3d[i])
            R = self.rotations[i]
            T = self.translations[i]

            newR, newT, diff_in_pix = remap_pose(self.K, R, T, pt3d, target_K, M)
            new_rotations.append(newR)
            new_translations.append(newT)
            
            # print(diff_in_pix)

        return PoseAnnot(
            self.keypoints_3d, target_K, new_masks, self.class_ids, 
            new_rotations, new_translations, target_width, target_height)

    def compute_keypoint_positions(self):
        obj_cnt = len(self.class_ids)
        kp_positions = []
        for i in range(obj_cnt):
            clsId = self.class_ids[i]
            R = self.rotations[i]
            T = self.translations[i]

            p3d = np.array(self.keypoints_3d[clsId])
            pts = np.matmul(self.K, np.matmul(R, p3d.transpose()) + T)
            xs = pts[0] / (pts[2] + 1e-8)
            ys = pts[1] / (pts[2] + 1e-8)
            kp_positions.append(np.concatenate((xs.reshape(-1,1),ys.reshape(-1,1)), axis=1))
        return np.stack(kp_positions)

    # def compute_dist_maps(self, mask, obj_cnt):
    #     dist_maps = []
    #     for i in range(obj_cnt):
    #         layer = np.ones_like(mask)
    #         layer[mask == (i+1)] = 0
    #         dmap = cv2.distanceTransform(layer, cv2.DIST_L2, 3)
    #         # if True:
    #         if False:
    #             tmpImg = cv2.normalize(layer, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    #             cv2.imshow("instance mask", tmpImg)
    #             tmpImg = cv2.normalize(dmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    #             cv2.imshow("dist_map", tmpImg)
    #             cv2.waitKey(0)
    #         dist_maps.append(dmap)
    #     return np.stack(dist_maps)

    def symmetry_handling(self, symmetry_types):
        # compute new RT for symmetry invariant handling
        if len(self.class_ids) == 0:
            return self

        new_rotations = []
        new_translations = []
        for i in range(len(self.class_ids)):
            cls_id = int(self.class_ids[i])
            R = self.rotations[i].numpy()
            T = self.translations[i].numpy()
            key_str = "cls_" + str(cls_id)
            if key_str in symmetry_types:
                R = pose_symmetry_handling(R, symmetry_types[key_str])
            R = torch.from_numpy(R)
            T = torch.from_numpy(T)
            new_rotations.append(R)
            new_translations.append(T)
        
        self.rotations = torch.stack(new_rotations)
        self.translations = torch.stack(new_translations)
        return self

    def visualize(self, cvImg):
        tmpPoses = self.to_numpy()
        cvImg = cvImg.copy()

        boxlist = tmpPoses.to_object_boxlist().bbox.tolist()
        # boxlist = tmpPoses.to_visible_boxlist().bbox.tolist()

        # tmpImg = cv2.normalize(tmpPoses.mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # cv2.imshow("maskImg", tmpImg)

        assert(len(boxlist) == len(tmpPoses.class_ids))
        for i in range(len(tmpPoses.class_ids)):
            bbox = boxlist[i]
            cvImg = cv2.rectangle(cvImg, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 0), 2)

            cls_id = int(tmpPoses.class_ids[i])
            R = tmpPoses.rotations[i]
            T = tmpPoses.translations[i]

            # GT keypoint reprojection
            pt3d = np.array(tmpPoses.keypoints_3d[cls_id])
            ptCnt = len(pt3d)
            pts = np.matmul(tmpPoses.K, np.matmul(R, pt3d.transpose()) + T)
            xs = pts[0] / (pts[2] + 1e-5)
            ys = pts[1] / (pts[2] + 1e-5)

            colors = plt.get_cmap('tab20', int(max(tmpPoses.class_ids)) + 1).colors * 255
            for pIdx in range(len(xs)):
                cvImg = cv2.circle(cvImg, (int(xs[pIdx]), int(ys[pIdx])), 3, colors[cls_id], -1)

            # draw pose axis
            # cvImg = draw_bounding_box(cvImg, R, T, pt3d, tmpPoses.K, (0,255,0))
            cvImg = draw_bounding_box(cvImg, R, T, pt3d, tmpPoses.K, (128,128,128))
            cvImg = draw_pose_axis(cvImg, R, T, pt3d, tmpPoses.K)

        return cvImg

    def remove_invalids(self, min_area=10):
        """
        check if segmentation masks have valid areas
        """
        new_classids = []
        new_rotations = []
        new_translations = []
        new_mask = torch.zeros_like(self.mask)
        curr_idx = 1
        valid_idx = []
        for i in range(len(self.class_ids)):
            tmpMask = (self.mask == i + 1)
            area = tmpMask.sum()
            if area < min_area:
                continue
            valid_idx.append(i)
            new_classids.append(self.class_ids[i])
            new_rotations.append(self.rotations[i])
            new_translations.append(self.translations[i])
            new_mask[tmpMask] = curr_idx
            curr_idx += 1

        if len(new_classids) > 0:
            self.class_ids = torch.stack(new_classids)
            self.rotations = torch.stack(new_rotations)
            self.translations = torch.stack(new_translations)
        else:
            self.class_ids = torch.LongTensor([])
            self.rotations = torch.FloatTensor([])
            self.translations = torch.FloatTensor([])
            
        self.mask = new_mask
        return self

    # Tensor-like methods
    def to_numpy(self):
        if isinstance(self.keypoints_3d, torch.Tensor):
            poses = PoseAnnot(
                self.keypoints_3d.numpy(),
                self.K.numpy(),
                self.mask.numpy(),
                self.class_ids.numpy(),
                self.rotations.numpy(),
                self.translations.numpy(),
                self.width, self.height
            )
            return poses
        else:
            return self

    def to_tensor(self):
        poses = PoseAnnot(
            torch.FloatTensor(self.keypoints_3d),
            torch.FloatTensor(self.K),
            torch.FloatTensor(self.mask),
            torch.LongTensor(self.class_ids),
            torch.FloatTensor(np.array(self.rotations)),
            torch.FloatTensor(np.array(self.translations)),
            self.width, self.height
            )
        return poses

    def to(self, device):
        poses = PoseAnnot(
            self.keypoints_3d.to(device), 
            self.K.to(device), 
            self.mask.to(device), 
            self.class_ids.to(device), 
            self.rotations.to(device), 
            self.translations.to(device), 
            self.width, self.height
            )
        return poses

    def __len__(self):
        return len(self.class_ids)

    def to_object_boxlist(self):
        # based on object 3D model, without considering occlusion
        objCnt = len(self.class_ids)
        bboxs = []
        for i in range(objCnt):
            if isinstance(self.mask, torch.Tensor):
                positions = (self.mask == (i+1)).nonzero(as_tuple=False)
                ys = positions[:, 0]
                xs = positions[:, 1]
            else:
                ys, xs = np.where(self.mask == (i+1))

            if len(xs) < 1:
                bboxs.append([0,0,0,0])
                continue

            # based on the reprojection of 3D bounding box
            clsId = self.class_ids[i]
            kp3d = self.keypoints_3d[clsId]
            R = self.rotations[i]
            T = self.translations[i]
            if isinstance(self.mask, torch.Tensor):
                reps = torch.matmul(self.K, torch.matmul(R, kp3d.t()) + T)
            else:
                reps = np.matmul(self.K, np.matmul(R, kp3d.transpose()) + T)
            xs = reps[0] / (reps[2] + 1e-8)
            ys = reps[1] / (reps[2] + 1e-8)
            bboxs.append([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())])
        if isinstance(self.mask, torch.Tensor):
            return BoxList(bboxs, (self.width, self.height), mode="xyxy").to(self.mask.device)
        else:
            return BoxList(bboxs, (self.width, self.height), mode="xyxy")

    def to_visible_boxlist(self):
        # based on masks
        objCnt = len(self.class_ids)
        bboxs = []
        for i in range(objCnt):
            if isinstance(self.mask, torch.Tensor):
                positions = (self.mask == (i+1)).nonzero(as_tuple=False)
                ys = positions[:, 0]
                xs = positions[:, 1]
            else:
                ys, xs = np.where(self.mask == (i+1))
            if len(xs) < 1:
                bboxs.append([0,0,0,0])
                continue
            bboxs.append([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())])
        if isinstance(self.mask, torch.Tensor):
            return BoxList(bboxs, (self.width, self.height), mode="xyxy").to(self.mask.device)
        else:
            return BoxList(bboxs, (self.width, self.height), mode="xyxy")

def test1():
    image_path = "/data/ycbv_light/train_real/000090/rgb/000001.jpg"
    bbox_json = "/data/ycbv_light/ycbv_bbox.json"
    meshpath = '/data/ycbv_light/models/'
    meshes, objID_2_clsID = load_bop_meshes(meshpath)
    K, merged_mask, class_ids, rotations, translations = \
        get_single_bop_annotation(image_path, objID_2_clsID)
    cvImg = cv2.imread(image_path)
    height, width, _ = cvImg.shape
    # 
    with open(bbox_json, 'r') as f:
        bbox_3d = np.array(json.load(f))
    # 
    poses = PoseAnnot(bbox_3d, K, merged_mask, class_ids, rotations, translations, width, height)
    poses.visualize(cvImg)

    M = generate_shiftscalerotate_matrix(0.2, 0.1, 15, width, height)

    # 
    target_width = 800
    target_height = 600
    target_K = np.array([[K[0][0], 0.0, target_width/2],
                        [0.0, K[1][1], target_height/2],
                        [0.0, 0.0, 1.0]], dtype=np.float32)
    # 
    cvImg = cv2.imread(image_path)
    cvImg = cv2.warpAffine(cvImg, M[:2], (target_width, target_height), flags=cv2.INTER_LINEAR, borderValue=(128, 128, 128))
    poses = poses.transform(M, target_K, target_width, target_height)
    cvImg = poses.visualize(cvImg)
    cv2.imshow("img", cvImg)
    cv2.waitKey(0)

def test2():
    image_path = "/data/ycbv_light/train_real/000090/rgb/000001.jpg"
    bbox_json = "/data/ycbv_light/ycbv_bbox.json"
    meshpath = '/data/ycbv_light/models/'
    meshes, objID_2_clsID = load_bop_meshes(meshpath)
    K, merged_mask, class_ids, rotations, translations = \
        get_single_bop_annotation(image_path, objID_2_clsID)
    cvImg = cv2.imread(image_path)
    height, width, _ = cvImg.shape
    # 
    with open(bbox_json, 'r') as f:
        bbox_3d = np.array(json.load(f))
    # 
    poses = PoseAnnot(bbox_3d, K, merged_mask, class_ids, rotations, translations, width, height)
    poses.visualize(cvImg)

    target_width = 320
    target_height = 240
    target_K = np.array([[K[0][0], 0.0, target_width/2],
                        [0.0, K[1][1], target_height/2],
                        [0.0, 0.0, 1.0]], dtype=np.float32)
    # 
    cx = width/2
    cy = height/2
    bw = width
    bh = height
    # 
    # resize the bounding box (keep image ratio)
    if (target_width/target_height) > (bw/bh):
        scale = target_height/bh
    else:
        scale = target_width/bw

    # move patch to image center and resize it to shape size (keeping ratio)
    M = np.array([[scale, 0.0, -scale*cx+target_width/2],
                    [0.0, scale, -scale*cy+target_height/2],
                    [0.0, 0.0, 1.0]], dtype=np.float32)
    # 
    cvImg = cv2.imread(image_path)
    cvImg = cv2.warpAffine(cvImg, M[:2], (target_width, target_height), flags=cv2.INTER_LINEAR, borderValue=(128, 128, 128))
    poses = poses.transform(M, target_K, target_width, target_height)
    cvImg = poses.visualize(cvImg)
    cv2.imshow("img", cvImg)
    cv2.waitKey(0)

if __name__ == "__main__":
    # test1()
    test2()
    pass
