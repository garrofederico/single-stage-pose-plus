import os
import cv2
import time

import sys
sys.path.append(os.path.abspath('./'))

from utils import *

def generate_one_image(cvImg, K, rotation, translation, pt3d, err_2d = 10):
    cellCnt = 20

    # GT keypoint reprojection
    pts = np.matmul(K, np.matmul(rotation, pt3d.transpose()) + translation)
    xs = pts[0] / pts[2]
    ys = pts[1] / pts[2]

    # repeat cellCnt times
    xs = xs.reshape(-1,1).repeat(cellCnt, axis=1)
    ys = ys.reshape(-1,1).repeat(cellCnt, axis=1)
    
    # add 2d err
    np.random.seed(0)
    noiseSigma = err_2d
    noisex = np.random.normal(0, noiseSigma, xs.shape).astype(np.float32)
    noisey = np.random.normal(0, noiseSigma, ys.shape).astype(np.float32)
    xs = xs + noisex
    ys = ys + noisey

    xy3d = pt3d.repeat(cellCnt, axis=0)
    xy2d = np.concatenate((xs.reshape(-1,1),ys.reshape(-1,1)), axis=1)

    # random order
    # if False:
    if True:
        np.random.seed()
        index = np.random.choice(np.arange(len(xy3d)), len(xy3d), replace=False)
        xy3d = xy3d[index]
        xy2d = xy2d[index]

    # compute pose
    retval, rot, trans, inliers = cv2.solvePnPRansac(xy3d, xy2d, K, None, flags=cv2.SOLVEPNP_EPNP)
    # retval, rot, trans = cv2.solvePnP(tmpv, xy2d, rawK, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if retval:
        print('%d/%d' % (len(inliers), len(xy2d)))
        R = cv2.Rodrigues(rot)[0]  # convert to rotation matrix
        T = trans.reshape(-1, 1)
        # cvImg = draw_pose_axis(cvImg, R, T, pt3d, K, 2)

    xs_flatten = xs.reshape(-1)
    ys_flatten = ys.reshape(-1)
    for pIdx in range(len(xs_flatten)):
        cvImg = cv2.circle(
            cvImg, 
            (int(xs_flatten[pIdx]), int(ys_flatten[pIdx])), 
            5, (0,255,0), -1, lineType=cv2.LINE_AA)

    # draw pose axis
    # cvImg = draw_bounding_box(cvImg, rotation, translation, pt3d, K, (0,255,0), 2)
    # cvImg = draw_pose_axis(cvImg, rotation, translation, pt3d, K, 2)

    # cv2.imshow('gtVisual', gtVisual)
    cv2.imshow('predicted', cvImg)
    cv2.waitKey(0)
    return cvImg

if __name__ == "__main__":
    img_path = '/data/ycbv_light/test/000052/rgb/000100.jpg'
    meshpath = '/data/ycbv_light/models/'
    bbox_json = '/data/ycbv_light/ycb_bbox.json'

    cls_id = 10 # pitcher box

    meshes, objID_2_clsID = load_bop_meshes(meshpath)
    with open(bbox_json, 'r') as f:
        keypoints_3d = json.load(f)

    mesh = meshes[cls_id]
    kp_3d = np.array(keypoints_3d[cls_id])

    cvImg = cv2.imread(img_path)

    height, width, _ = cvImg.shape
    K, merged_mask, class_ids, rotations, translations = get_single_bop_annotation(img_path, objID_2_clsID)

    ins_id = 2
    out = cv2.VideoWriter('out.mp4',cv2.VideoWriter_fourcc('X','2','6','4'), 10, (width, height))
    for i in range(100):
        tmpImg = np.copy(cvImg)
        tmpImg = generate_one_image(tmpImg, K, rotations[ins_id], translations[ins_id], kp_3d)
        cv2.imshow("img", tmpImg)
        cv2.waitKey(100)
        out.write(tmpImg)
    out.release()