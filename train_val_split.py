from __future__ import division, print_function

import glob
import os
import shutil
import numpy as np

def train_val_split(kfold=7, fold_id=0):

    val_ratio = 1/kfold

    np.random.seed(42)
    src_root = '/home/omewangyx/kaggle/carvical_cancer/zhouyusong_code/data/train_add2_seg'
    dst_root = '/home/omewangyx/kaggle/carvical_cancer/zhouyusong_code/data/train_data_path'

    classes = [d for d in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}

    val_dir = os.path.join(dst_root, 'val')
    train_dir = os.path.join(dst_root, 'train')
    for ix in range(3):
        class_dir = os.path.join(val_dir, classes[ix])
        if os.path.exists(class_dir):
            shutil.rmtree(class_dir)
        class_dir = os.path.join(train_dir, classes[ix])
        if os.path.exists(class_dir):
            shutil.rmtree(class_dir)

    train_imgs = list([glob.glob(os.path.join(src_root, x, '*.jpg')) for x in classes])
    num_imgs = tuple((len(x) for x in train_imgs))
    # print('There are {} classes with {} images each'.format(len(classes), num_imgs))

    for ix, imgs in enumerate(train_imgs[:]):
        np.random.shuffle(imgs)
        val_num = int(val_ratio * len(imgs))
        val_set = imgs[val_num*fold_id:val_num*(fold_id+1)]
        train_set  = imgs[val_num*(fold_id+1):] + imgs[:val_num*fold_id]
        
        class_dir = os.path.join(val_dir, classes[ix])
        if not os.path.exists(class_dir):
            os.mkdir(class_dir)
        for im in val_set:
            shutil.copy(im, class_dir)
        
        class_dir = os.path.join(train_dir, classes[ix])
        if not os.path.exists(class_dir):
            os.mkdir(class_dir)
        for im in train_set:
            shutil.copy(im, class_dir)

def train_val_split_bag(max_sample=0.8):

    src_root = '/home/omewangyx/kaggle/carvical_cancer/liufei_code/data/train_add_seg'
    dst_root = '/home/omewangyx/kaggle/carvical_cancer/liufei_code/data/add_seg_bag'

    classes = [d for d in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}

    val_dir = os.path.join(dst_root, 'val')
    train_dir = os.path.join(dst_root, 'train')
    for ix in range(3):
        class_dir = os.path.join(val_dir, classes[ix])
        if os.path.exists(class_dir):
            shutil.rmtree(class_dir)
        class_dir = os.path.join(train_dir, classes[ix])
        if os.path.exists(class_dir):
            shutil.rmtree(class_dir)

    train_imgs = list([glob.glob(os.path.join(src_root, x, '*.jpg')) for x in classes])
    num_imgs = tuple((len(x) for x in train_imgs))
    # print('There are {} classes with {} images each'.format(len(classes), num_imgs))

    for ix, imgs in enumerate(train_imgs[:]):
        train_num = int(max_sample * len(imgs))
        train_set = np.random.choice(imgs, train_num)
        val_set = list(set(imgs).difference(set(train_set)))
        print('Train set size: {}  Validation set size: {}'.format(len(train_set), len(val_set)))
            
        class_dir = os.path.join(train_dir, classes[ix])
        if not os.path.exists(class_dir):
            os.mkdir(class_dir)
        for im in train_set:
            im_name, extension = im.split('/')[-1].split('.')
            if not os.path.exists(os.path.join(class_dir, im_name+'.'+extension)):
                shutil.copy(im, class_dir)
            else:
                ii = 1
                while True:
                    new_im = os.path.join(class_dir, im_name + '_' + str(ii) + '.' + extension)
                    if not os.path.exists(new_im):
                        shutil.copy(im, new_im)
                        break
                    ii += 1
                    
        
        class_dir = os.path.join(val_dir, classes[ix])
        if not os.path.exists(class_dir):
            os.mkdir(class_dir)
        for im in val_set:
            shutil.copy(im, class_dir)
