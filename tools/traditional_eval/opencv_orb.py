import os

import numpy as np
import cv2

import sys
sys.path.append(os.path.abspath('./'))

from utils import (
    load_bop_meshes,
    get_single_bop_annotation,
    render_objects
)

def get_model_descriptors(imag_path, mesh, objID_2_clsID):
    K, merged_mask, class_ids, rotations, translations = \
        get_single_bop_annotation(image_path, objID_2_clsID)
    cvImg = cv2.imread(image_path)
    height, width, _ = cvImg.shape

    pose = np.concatenate((rotations[0],translations[0]), axis=1)
    rImg, rDepth = render_objects([mesh], [0], [pose], K, width, height)

    cv2.imshow("rimg", rImg)
    cv2.imshow("depth", rDepth)
    # cv2.waitKey(0)

    orb = cv2.ORB_create(nfeatures=1000, scoreType=cv2.ORB_FAST_SCORE)
    # find the keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(cvImg, None)

    tmpImg = cv2.drawKeypoints(
        cvImg,kp1,None, color=(0,255,0),
        # flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    cv2.imshow("keypoint1", tmpImg)

    cv2.imshow("img", cvImg)
    cv2.waitKey(0)

# refer: https://stackoverflow.com/questions/44719630/real-time-pose-estimation-of-a-textured-object

if __name__ == "__main__":

    image_path = "/data/vespa_syn_0327/train/000000/rgb/015852.png"
    bbox_json = "/data/vespa_syn_0327/vespa_bbox.json"
    meshpath = '/data/vespa_syn_0327/models/'
    meshes, objID_2_clsID = load_bop_meshes(meshpath)
    get_model_descriptors(image_path, meshes[0], objID_2_clsID)

    # image_path1 = "/data/vespa_syn_0327/train/000000/rgb/015852.png"
    # image_path2 = "/data/vespa_syn_0327/train/000000/rgb/015967.png"

    image_path1 = "/data/ycbv_light/train_real/000020/rgb/000001.jpg"
    image_path2 = "/data/ycbv_light/train_real/000020/rgb/001400.jpg"

    # image_path1 = '/data/speed_aux/train/000000/rgb/000017.jpg'
    # image_path2 = '/data/speed_aux/train/000000/rgb/000020.jpg'

    img1 = cv2.imread(image_path1) # queryImage
    img2 = cv2.imread(image_path2) # trainImage

    # cv2.imshow("img1", img1)
    # cv2.imshow("img2", img2)

    # Initiate detector
    orb = cv2.ORB_create()

    # find the keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    tmpImg = cv2.drawKeypoints(
        img1,kp1,None, color=(0,255,0),
        # flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    cv2.imshow("keypoint1", tmpImg)

    tmpImg = cv2.drawKeypoints(
        img2,kp2,None, color=(0,255,0),
        # flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    cv2.imshow("keypoint2", tmpImg)

    # FLANN parameters
    FLANN_INDEX_LSH = 6
    index_params= dict(algorithm = FLANN_INDEX_LSH,
                    table_number = 6, # 12
                    key_size = 12,     # 20
                    multi_probe_level = 1) #2
    search_params = dict(checks=100)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # if False:
    if True:
        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in range(len(matches))]

        # ratio test as per Lowe's paper
        for i, match in enumerate(matches):
            if len(match) < 2:
                continue
            m = match[0]
            n = match[1]
            if m.distance < 0.7*n.distance:
                matchesMask[i]=[1,0]

    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = 2)

    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

    cv2.imshow("matching", img3)
    cv2.waitKey(0)
