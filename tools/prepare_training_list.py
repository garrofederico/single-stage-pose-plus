import os
import random

import sys
sys.path.append(os.path.abspath('./'))

from utils import collect_bop_imagelist

if __name__ == "__main__":
    training_dir1 = '/home/yhu/ycbv/train_real/'
    training_dir2 = '/home/yhu/ycbv/train_synt/'
    out_dir = "/home/yhu/ycbv/"

    training_list1 = collect_bop_imagelist(training_dir1)
    training_list2 = collect_bop_imagelist(training_dir2)
    
    random.shuffle(training_list1)
    validation_list = training_list1[:5000]
    validation_list.sort()
    training_list = training_list1[5000:]
    training_list.sort() 
    training_list += training_list2

    with open(out_dir + 'train.txt', 'w') as f:
        for itm in training_list:
            f.write(itm + '\n')
    
    with open(out_dir + 'valid.txt', 'w') as f:
        for itm in validation_list:
            f.write(itm + '\n')