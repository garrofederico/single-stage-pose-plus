import os

import sys
sys.path.append(os.path.abspath('./'))

from utils import collect_mesh_bbox

if __name__ == "__main__":
    # outdir = '/data/linemod/'
    # meshpath = '/data/linemod/models/'
    outdir = '/data/ycbv_light/'
    meshpath = '/data/ycbv_light/models/'
    collect_mesh_bbox(meshpath,  outdir + 'ycbv_bbox.json', oriented=True)
    # outdir = '/data/speed_aux/'
    # meshpath = '/data/speed_aux/models/'
    # outdir = '/data/vespa_syn_0327/'
    # meshpath = '/data/vespa_syn_0327/models/'
    # outdir = '/data/swisscube_syn_0329/'
    # meshpath = '/data/swisscube_syn_0329/models/'
    # collect_mesh_bbox(meshpath,  outdir + 'swisscube_bbox.json', oriented=False)