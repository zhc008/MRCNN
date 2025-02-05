#!/usr/bin/env python
# Copyright 2019 Division of Medical Image Computing, German Cancer Research Center (DKFZ).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

'''
This preprocessing script loads nrrd files obtained by the data conversion tool: https://github.com/MIC-DKFZ/LIDC-IDRI-processing/tree/v1.0.1
After applying preprocessing, images are saved as numpy arrays and the meta information for the corresponding patient is stored
as a line in the dataframe saved as info_df.pickle.
'''

import os
import sys
import time
import subprocess

from skimage import io as skio
import numpy as np
import numpy.testing as npt
from skimage.transform import resize
import pandas as pd
from scipy import ndimage

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append('../..')
import data_manager as dmanager

class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def resample_array(src_imgs, src_spacing, target_spacing):
    """ Resample a numpy array.
    :param src_imgs: source image.
    :param src_spacing: source image's spacing.
    :param target_spacing: spacing to resample source image to.
    :return:
    """
    src_spacing = np.round(src_spacing, 3)
    target_shape = [int(src_imgs.shape[ix] * src_spacing[::-1][ix] / target_spacing[::-1][ix]) for ix in range(len(src_imgs.shape))]
    for i in range(len(target_shape)):
        try:
            assert target_shape[i] > 0
        except:
            raise AssertionError("AssertionError:", src_imgs.shape, src_spacing, target_spacing)

    img = src_imgs.astype('float64')
    resampled_img = resize(img, target_shape, order=1, clip=True, mode='edge').astype('float32')

    return resampled_img

# make all the labels in the image to be labelled from 1 to number of labels
def reindex_labels(im):
    labels = np.unique(im)
    for i in range(labels.shape[0]):
        im[im == labels[i]] = i

# crop an image to 5 smaller ones
def crop_image(raw_dir, raw_save_dir, label_dir, label_save_dir):
    # the size and stride of the crop
    size = 96
    depth = 32
    stride = 66
    z_stride = 12
    # size = 64
    # depth = 48
    # stride = 20
    # z_stride = 31
    # size = 64
    # depth = 48
    # stride = 44
    # z_stride = 31
    # size = 64
    # depth = 32
    # stride = 44
    # z_stride = 12
    # size = 64
    # depth = 16
    # stride = 44
    # z_stride = 6

    # read the raw image in
    raw = skio.imread(raw_dir)
    raw_t = raw.transpose((1,2,0)) # transpose the image to fit the model
    raw_t_cut = raw_t[:,:,20:-20] # cut out the first and last 20 slices

    # read the labelled image in
    label = skio.imread(label_dir)
    label_t = label.transpose((1,2,0))
    label_t_cut = label_t[:,:,20:-20]

    # crop images into smaller patches
    height, width, zidth = raw_t.shape
    # print(raw_t.shape)
    sub_imblack = np.copy(raw_t_cut)[:size, :size, :depth]
    sub_imblack[sub_imblack > 0] = 0
    skio.imsave(raw_save_dir + "/crop0.tif", arr = sub_imblack) # save a black raw image
    skio.imsave(label_save_dir + "/crop0.tif", arr = sub_imblack) # save a black label image
    sub_index = 1
    from collections import Counter
    count = []
    for y in range(0, height - size + 1, stride):
        for x in range(0, width - size + 1, stride):
            for z in range(0, zidth - depth + 1, z_stride):
                raw_temp = np.copy(raw_t_cut)[y:y+size, x:x+size, z:z+depth]
                label_temp = np.copy(label_t_cut)[y:y+size, x:x+size, z:z+depth]
                # print(np.unique(label_temp).shape[0])
                # count.append(np.unique(label_temp).shape[0])
                # skip volume with less than 97 synapses
                # if np.unique(label_temp).shape[0] <= 97:      # 64 64 48
                # if np.unique(label_temp).shape[0] <= 20:      # 32 32 32
                if np.unique(label_temp).shape[0] <= 170:     # 96 96 32
                # if np.unique(label_temp).shape[0] <= 85:        # 64 64 32
                # if np.unique(label_temp).shape[0] <= 35:        # 64 64 16
                    continue
                # # change volume with less than 100 synapses to black volume
                # elif np.unique(label_temp).shape[0] <= 100:   # 64 64 48
                # # elif np.unique(label_temp).shape[0] <= 21:  # 32 32 32
                elif np.unique(label_temp).shape[0] <= 174:   # 96 96 32
                # elif np.unique(label_temp).shape[0] <= 87:      # 64 64 32
                # elif np.unique(label_temp).shape[0] <= 37:      # 64 64 16
                    raw_temp[raw_temp > 0] = 0
                    raw_temp[raw_temp < 0] = 0
                    label_temp[label_temp > 0] = 0
                # elif np.unique(label_temp).shape[0] > 380:
                #     continue
                skio.imsave(raw_save_dir + "/crop" + str(sub_index) + ".tif", arr = raw_temp) # save raw images
                skio.imsave(label_save_dir + "/crop" + str(sub_index) + ".tif", arr = label_temp) # save label images
                
                # if sub_index > 450:
                if sub_index > 100:
                    break
                sub_index += 1
    # count.sort()
    # print(Counter(count).keys())
    # print(Counter(count).values())


def preprocess_image(cf):
    # crop_image(cf.before_crop_dir + "/rsc01_reg_XTC_t1.tif", cf.raw_data_dir, cf.before_crop_dir + "/rsc01_reg_XTC_t1_segmented.tif", cf.edited_label_data_dir)
    # raw_files = os.listdir(cf.raw_data_dir)
    raw_files = os.listdir(cf.blacken_raw_dir)
    file_number = len(raw_files)
    for i in range(file_number):
        file_name = 'crop' + str(i) + '.tif'
        raw = skio.imread(os.path.join(cf.blacken_raw_dir, file_name)).astype(np.uint32)
        out_path = os.path.join(cf.pp_dir, '{}.npy'.format(i))
        np.save(out_path, raw)
    # label_files = os.listdir(cf.edited_label_data_dir)
    label_files = os.listdir(cf.blacken_label_dir)
    file_number = len(label_files)
    for i in range(file_number):
        # file_name = 'crop' + str(i) + '_processed.tif'
        file_name = 'crop' + str(i) + '.tif'
        label = skio.imread(os.path.join(cf.blacken_label_dir, file_name)).astype(np.uint32) # 32 by 96 by 96
        # label = label.transpose((1,2,0)) # transpose to 96 by 96 by 32
        reindex_labels(label)
        out_path_seg = os.path.join(cf.pp_dir, '{}_seg.npy'.format(i))
        np.save(out_path_seg, label)

def create_dataframe(cf):
    label_files = os.listdir(cf.blacken_label_dir)
    file_number = len(label_files)
    class_ids = np.array(np.ones(((file_number,1)))).astype(int)
    pid = np.linspace(0,file_number-1, file_number)
    pid = pid[:,None].astype(int)
    df_data = np.concatenate((class_ids,pid), axis=1)
    # class_targets = list(np.linspace(1,760,760).astype(int))
    # class_targets = list([1])
    df = pd.DataFrame(df_data,columns=['class_ids', 'pid'])
    fg_slices = np.zeros((file_number, 1))
    df['fg_slices'] = fg_slices.astype(object)
    df['class_ids'] = df['class_ids'].astype(object)
    df['regression_vectors'] = ""
    df['undistorted_rg_vectors'] = ""
    # for i in range(len(df['class_ids'])):
    #     # set class_ids to a list of 1
    #     df.at[i,'class_ids'] = class_targets
        ## the black ones
        # if i == 0 or i == len(df['class_ids'])-1:
        #     df.at[i,'class_ids'] = []

    for i in range(file_number):
        # file_name = 'crop' + str(i) + '_processed.tif'
        file_name = 'crop' + str(i) + '.tif'
        label = skio.imread(os.path.join(cf.blacken_label_dir, file_name)).astype(np.uint32)
        df.at[i,'class_ids'] = [1] * np.unique(label).shape[0]
        ## the black ones
        # if i == 0 or i == len(df['class_ids'])-1:
        if np.unique(label).shape[0] == 1:
            df.at[i,'class_ids'] = []
        # set fg_slices
        df.at[i, 'fg_slices'] = list(np.arange(0,label.shape[2]))

    npz_dir = os.path.join(cf.pp_dir+'_npz')
    df.to_pickle(os.path.join(npz_dir, 'info_df.pickle'))
    df.to_pickle(os.path.join(cf.pp_dir, 'info_df.pickle'))


def convert_copy_npz(cf):
    npz_dir = os.path.join(cf.pp_dir+'_npz')
    print("converting to npz dir", npz_dir)
    os.makedirs(npz_dir, exist_ok=True)

    dmanager.pack_dataset(cf.pp_dir, destination=npz_dir, recursive=True, verbose=False)
    subprocess.call('rsync -avh --exclude="*.npy" {} {}'.format(cf.pp_dir, npz_dir), shell=True)

def augment_blacken(cf):
    raw_files = os.listdir(cf.raw_data_dir)
    file_number = len(raw_files)
    for i in range(file_number):
        file_name = 'crop' + str(i) + '.tif'
        raw = skio.imread(os.path.join(cf.raw_data_dir, file_name)).astype(np.uint32)
        augment_raw = raw.copy()
        # print(f"augment_raw.shape: {augment_raw.shape}")
        augment_raw[:,:1,:] = 0
        out_path = os.path.join(cf.blacken_raw_dir, 'crop{}.tif'.format(i*2))
        augment_path = os.path.join(cf.blacken_raw_dir, 'crop{}.tif'.format(i*2 + 1))
        skio.imsave(out_path, arr=raw)
        skio.imsave(augment_path, arr=augment_raw)
    label_files = os.listdir(cf.edited_label_data_dir)
    file_number = len(label_files)
    for i in range(file_number):
        file_name = 'crop' + str(i) + '_processed.tif'
        label = skio.imread(os.path.join(cf.edited_label_data_dir, file_name)).astype(np.uint32) # 32 by 96 by 96
        label = label.transpose((1,2,0)) # transpose to 96 by 96 by 32
        augment_label = label.copy()
        augment_label[:,:1,:] = 0
        # reindex_labels(label)
        out_path_seg = os.path.join(cf.blacken_label_dir, 'crop{}.tif'.format(i*2))
        augment_path_seg = os.path.join(cf.blacken_label_dir, 'crop{}.tif'.format(i*2 + 1))
        skio.imsave(out_path_seg, arr=label)
        skio.imsave(augment_path_seg, arr=augment_label)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-n', '--number', type=int, default=None, help='How many patients to maximally process.')
    # args = parser.parse_args()
    total_stime = time.time()

    import configs
    cf = configs.Configs()

    augment_blacken(cf)
    # preprocess the image from TIFF to .npy
    preprocess_image(cf)
    # convert npy files' copy to npz file
    convert_copy_npz(cf)

    create_dataframe(cf)

    mins, secs = divmod((time.time() - total_stime), 60)
    h, mins = divmod(mins, 60)
    t = "{:d}h:{:02d}m:{:02d}s".format(int(h), int(mins), int(secs))
    print("{} total runtime: {}".format(os.path.split(__file__)[1], t))
