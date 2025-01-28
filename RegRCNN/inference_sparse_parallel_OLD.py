"""for presentations etc"""

from cProfile import label
from inspect import cleandoc
import plotting as plg
import cProfile
import io
import pstats

import sys
import os
import pickle

import numpy as np
import pandas as pd
import torch

import utils.exp_utils as utils
import utils.model_utils as mutils
from predictor import Predictor
from evaluator import Evaluator

import matplotlib.pyplot as plt
import tifffile as tiff
import skimage.io as skio
from tqdm import tqdm
# from skimage.measure import regionprops
from skimage import measure
from skimage.exposure import match_histograms
from torch.utils.data import Dataset
import torch
import multiprocessing
from functools import partial
import time
from scipy.sparse import coo_matrix, hstack, lil_matrix
from scipy import sparse
from scipy.sparse.csgraph import connected_components


""" Tiger function """
def plot_max(im, ax=0, plot=1):
     max_im = np.amax(im, axis=ax)
     if plot:
         plt.figure(); plt.imshow(max_im)

     return max_im



""" Tiger function """
def boxes_to_mask(cf, results_dict, thresh, input_im, unique_id=0):

    label_arr = np.copy(results_dict['seg_preds'],)
    new_labels = np.zeros(np.shape(results_dict['seg_preds']))
    class_labels = np.zeros(np.shape(results_dict['seg_preds']))
    check_box = np.zeros(np.shape(results_dict['seg_preds']))
    # print(np.unique(label_arr))
    # print(label_arr.shape)
    for box_id, box_row in enumerate(results_dict['boxes'][0]):

        print('box_score: ' + str(box_row['box_score']))


        #if cf.dim == 2 and box_row['box_score'] < cf.merge_3D_iou:
        #    continue
        #else:

        if box_row['box_score'] >= thresh:

            box_arr = np.zeros(np.shape(results_dict['seg_preds']))
            bc = box_row['box_coords']
            bc = np.asarray(bc, dtype=np.int)

            bc[np.where(bc < 0)[0]] = 0  ### cannot be negative

            ### also cannot be larger than image size
            if cf.dim == 2:
                bc[np.where(bc >= label_arr.shape[-1])[0]] = label_arr.shape[-1]

                box_arr[bc[4]:bc[5], 0, bc[0]:bc[2], bc[1]:bc[3]] = box_id + 1    ### +1 because starts from 0
                box_arr[label_arr == 0] = 0

                new_labels[box_arr > 0] = box_id + 1

                class_labels[box_arr > 0] = box_row['box_pred_class_id']

            else:
                bc[0:4][np.where(bc[0:4] >= label_arr.shape[-2])[0]] = label_arr.shape[-2]

                bc[4:6][np.where(bc[4:6] >= label_arr.shape[-1])[0]] = label_arr.shape[-1]

                box_arr[0, 0, bc[0]:bc[2], bc[1]:bc[3], bc[4]:bc[5],] = box_id + 1    ### +1 because starts from 0
                # box_arr[0, 0, bc[0]:bc[2], bc[1]:bc[3], bc[4]:bc[5],][new_labels[0, 0, bc[0]:bc[2], bc[1]:bc[3], bc[4]:bc[5],
                #                                                                  ] == 0] = box_id + 1    ### +1 because starts from 0
                # check_box[box_arr > 0] = box_row['box_pred_class_id']
                box_arr[label_arr == 0] = 0
                # box_arr[input_im < 12] = 0 # added to avoid segmentation that includes background
                new_labels[box_arr > 0] = box_id + 1


                class_labels[box_arr > 0] = box_row['box_pred_class_id']

        # added print statements
        # print(label_arr.shape)
        # print(class_labels.shape)

    label_arr = new_labels
    # np.save('/cis/home/zchen163/my_documents/debugging/check_box', np.squeeze(check_box))

    return label_arr, class_labels





def find_pid_in_splits(pid, exp_dir=None):
    if exp_dir is None:
        exp_dir = cf.exp_dir
    check_file = os.path.join(exp_dir, 'fold_ids.pickle')
    with open(check_file, 'rb') as handle:
        splits = pickle.load(handle)

    finds = []
    for i, split in enumerate(splits):
        if pid in split:
            finds.append(i)
            print("Pid {} found in split {}".format(pid, i))
    if not len(finds)==1:
        raise Exception("pid {} found in more than one split: {}".format(pid, finds))
    return finds[0]







def plot_train_forward(slices=None):
    with torch.no_grad():
        batch = next(val_gen)

        results_dict = net.train_forward(batch, is_validation=True) #seg preds are int preds already
        print(results_dict['seg_preds'].shape)
        print(batch['data'].shape)


        out_file = os.path.join(anal_dir, "straight_val_inference_fold_{}".format(str(cf.fold)))
        #plg.view_batch(cf, batch, res_dict=results_dict, show_info=False, legend=True,
        #                          out_file=out_file)#, slices=slices)

        ### TIGER - SAVE AS TIFF
        truth_im = np.expand_dims(batch['seg'], axis=0)


        ### if 3D
        #seg_im = np.moveaxis(results_dict['seg_preds'], -1, 1)
        import tifffile as tiff
        tiff.imwrite(out_file + '_TRUTH.tif', np.asarray(truth_im, dtype=np.uint32),
                      imagej=True,   metadata={'spacing': 1, 'unit': 'um', 'axes': 'TZCYX'})


        seg_im = np.expand_dims(results_dict['seg_preds'], axis=0)
        ### if 3D
        #seg_im = np.moveaxis(results_dict['seg_preds'], -1, 1)
        import tifffile as tiff
        tiff.imwrite(out_file + '_seg.tif', np.asarray(seg_im, dtype=np.uint32),
                      imagej=True,   metadata={'spacing': 1, 'unit': 'um', 'axes': 'TZCYX'})


        input_im = np.expand_dims(batch['data'], axis=0)


        ### if 3D
        #input_im = np.moveaxis(batch['data'], -1, 1)
        tiff.imwrite(out_file + '_input_im.tif', np.asarray(input_im, dtype=np.uint32),
                      imagej=True,   metadata={'spacing': 1, 'unit': 'um', 'axes': 'TZCYX'})




        ### TIGER ADDED
        import utils.exp_utils as utils
        print('Plotting output')
        utils.split_off_process(plg.view_batch, cf, batch, results_dict, has_colorchannels=cf.has_colorchannels,
                                show_gt_labels=True, get_time="val-example plot",
                                out_file=os.path.join(cf.plot_dir, 'batch_example_val_{}.png'.format(cf.fold)))


def plot_forward(pid, slices=None):
    with torch.no_grad():
        batch = batch_gen['test'].generate_train_batch(pid=pid)
        results_dict = net.test_forward(batch) #seg preds are only seg_logits! need to take argmax.

        if 'seg_preds' in results_dict.keys():
            results_dict['seg_preds'] = np.argmax(results_dict['seg_preds'], axis=1)[:,np.newaxis]

        out_file = os.path.join(anal_dir, "straight_inference_fold_{}_pid_{}".format(str(cf.fold), pid))




        print(results_dict['seg_preds'].shape)
        print(batch['data'].shape)

        ### TIGER - SAVE AS TIFF
        truth_im = np.expand_dims(batch['seg'], axis=0)


        ### if 3D
        #seg_im = np.moveaxis(results_dict['seg_preds'], -1, 1)
        import tifffile as tiff
        tiff.imwrite(out_file + '_TRUTH.tif', np.asarray(truth_im, dtype=np.uint32),
                      imagej=True,   metadata={'spacing': 1, 'unit': 'um', 'axes': 'TZCYX'})



        roi_mask = np.expand_dims(batch['roi_masks'][0], axis=0)


        ### if 3D
        #seg_im = np.moveaxis(results_dict['seg_preds'], -1, 1)
        import tifffile as tiff
        tiff.imwrite(out_file + '_roi_mask.tif', np.asarray(roi_mask, dtype=np.uint32),
                      imagej=True,   metadata={'spacing': 1, 'unit': 'um', 'axes': 'TZCYX'})




        seg_im = np.expand_dims(results_dict['seg_preds'], axis=0)


        ### if 3D
        #seg_im = np.moveaxis(results_dict['seg_preds'], -1, 1)
        import tifffile as tiff
        tiff.imwrite(out_file + '_seg.tif', np.asarray(seg_im, dtype=np.uint32),
                      imagej=True,   metadata={'spacing': 1, 'unit': 'um', 'axes': 'TZCYX'})


        input_im = np.expand_dims(batch['data'], axis=0)


        ### if 3D
        #input_im = np.moveaxis(batch['data'], -1, 1)
        tiff.imwrite(out_file + '_input_im.tif', np.asarray(input_im, dtype=np.uint32),
                      imagej=True,   metadata={'spacing': 1, 'unit': 'um', 'axes': 'TZCYX'})




        ### This below hangs

        # plg.view_batch(cf, batch, res_dict=results_dict, show_info=False, legend=True, show_gt_labels=True,
        #                           out_file=out_file, sample_picks=slices, has_colorchannels=False)

        print('Plotting output')
        utils.split_off_process(plg.view_batch, cf, batch, results_dict, has_colorchannels=cf.has_colorchannels,
                                show_gt_labels=True, get_time="val-example plot",
                                out_file=os.path.join(cf.plot_dir, 'batch_SINGLE_PID_{}.png'.format(pid)))






def plot_merged_boxes(results_list, pid, plot_mods=False, show_seg_ids="all", show_info=True, show_gt_boxes=True,
                      s_picks=None, vol_slice_picks=None, score_thres=None):
    """

    :param results_list: holds (results_dict, pid)
    :param pid:
    :return:
    """
    results_dict = [res_dict for (res_dict, pid_) in results_list if pid_==pid][0]
    #seg preds are discarded in predictor pipeline.
    #del results_dict['seg_preds']

    batch = batch_gen['test'].generate_train_batch(pid=pid)
    out_file = os.path.join(anal_dir, "merged_boxes_fold_{}_pid_{}_thres_{}.png".format(str(cf.fold), pid, str(score_thres).replace(".","_")))

    utils.save_obj({'res_dict':results_dict, 'batch':batch}, os.path.join(anal_dir, "bytes_merged_boxes_fold_{}_pid_{}".format(str(cf.fold), pid)))

    plg.view_batch(cf, batch, res_dict=results_dict, show_info=show_info, legend=False, sample_picks=s_picks,
                   show_seg_pred=True, show_seg_ids=show_seg_ids, show_gt_boxes=show_gt_boxes,
                   box_score_thres=score_thres, vol_slice_picks=vol_slice_picks, show_gt_labels=True,
                   plot_mods=plot_mods, out_file=out_file, has_colorchannels=cf.has_colorchannels, dpi=600)

    return


def padding(im, pad_size=8, axis=0):
    padded_size = im.shape[axis] + 2 * pad_size
    output = np.zeros((padded_size, im.shape[1], im.shape[2]), dtype=im.dtype)
    output[pad_size:-pad_size,:,:] = im
    return output

def reindex_labels(im):
    labels = np.unique(im)
    for i in range(labels.shape[0]):
        im[im == labels[i]] = i

def edit_labels(waterim):
    if np.max(waterim)==0:
        return waterim

    from skimage import measure
    import pandas as pd
    waterim=np.transpose(waterim,axes=[2,0,1])
    # print(f"waterim.shape: {waterim.shape}")

    waterim=measure.label(waterim, connectivity=1) #due to the cutting of the large ilastik file to create training patches,some segs were cut up and are not unified anymore
    # print(f"len(waterim): {len(waterim)}")

    #%% STEPS 1,3&4 TESTING DOING IT by slice instead, it's relatively fast on the small patches, less than one minute to run 100 patches, but much slower for the real sized images

    smalldeletions=[]

    for zz in range(len(waterim)): #this step takes a ~2 mins per 1024x1024x181 image
        zim = measure.label(waterim[zz],connectivity=1)
        # print(np.unique(zim))
        if np.max(zim) == 0:
            continue
        zprops = pd.DataFrame(measure.regionprops_table(zim, properties=('label', 'area','coords','eccentricity')))

        #Replacing step 1 and combining with step 3
        smallsegs = zprops['coords'][zprops['area'] <= 3].copy()

        for co in smallsegs:
            for xycoord in co:
                zlocation=[zz]
                zlocation.extend(xycoord)
                smalldeletions.append(zlocation)

        #For removing linear segs
        linears = zprops['coords'][zprops['eccentricity'] ==1].copy()
        for co in linears:
            for xycoord in co:
                zlocation=[zz]
                zlocation.extend(xycoord)
                if zlocation not in smalldeletions:
                    smalldeletions.append(zlocation)

    for sm in smalldeletions:
        waterim[sm[0],sm[1],sm[2]] = 0 #This step is instantaneous even with 60k pixels to remove

    #%% Instead of removing hollow synapses, just rerun labeling with conn=1
    waterim=measure.label(waterim, connectivity=1)
    if np.max(waterim) == 0:
        waterim=np.asarray(waterim,dtype=np.uint32)
        waterim=np.transpose(waterim,axes=[1,2,0])
        return waterim

    feats=pd.DataFrame(measure.regionprops_table(waterim, properties=('label', 'area','coords')))
    feats['zplanes']=feats['coords'].apply(lambda x: np.unique(x[:,0]).shape[0])

    zlessthan3=list(np.asarray(feats.loc[feats['zplanes'] <= 2])[:,0])
    waterim[np.isin(waterim,zlessthan3)]=0

    #%% relabel and save

    waterim=measure.label(waterim, connectivity=1)
    waterim=np.asarray(waterim,dtype=np.uint32)
    waterim=np.transpose(waterim,axes=[1,2,0])
    return waterim

def patch_inference(input): #Function for dense merging don't change this (not used anymore since sparse works)
    quad_size = 96
    quad_depth = 32
    stride = 62 # 1024*1024*303
    # stride = 51  # 300*300*180
    z_stride = 16 # 1024*1024*303

    width=344
    height=344
    depth_im=128
    z_ind = input%3
    y_ind = int(np.floor(input/3))%4
    x_ind = int(np.floor(input / (4 * 3)))


    segmentation = np.zeros([depth_im, width, height])
    # for i in range(len(input)):
    #     cleaned_seg, coord = input[i]
    #     cleaned_seg = cleaned_seg.squeeze()
    #     z, y, x = coord
    buffer_dir = '/cis/home/zchen163/my_documents/XTC_data/RSC03_roi1_buffer/'
    memmap_dir = '/cis/home/zchen163/my_documents/XTC_data/RSC03_roi1_memmap/'
    for x in range(4):
        if x_ind == 3 and x == 3:
            x = x*stride-3
        else:
            x = x * stride
        for y in range(4):
            if y_ind == 3 and y == 3:
                y = y*stride-3
            else:
                y = y * stride
            for z in range(6):
                if z_ind == 2 and z == 5:
                    z = z * z_stride - 2
                else:
                    z = z * z_stride
                # cleaned_seg = np.load(buffer_dir + f'patches_after_inference/p{input}_z{z}_x{x}_y{y}.npy')
                temp_memmap = np.memmap(memmap_dir + f'patches_after_inference/p{input}_z{z}_x{x}_y{y}'
                                        , dtype='uint32', mode='r', shape=(quad_depth, quad_size, quad_size))
                cleaned_seg = np.array(temp_memmap)
                del temp_memmap
                # skio.imsave(sav_dir + f"mask_x{x}_y{y}_z{z}.tif", arr=cleaned_seg)
                # break
                max_label = np.max(segmentation)
                cleaned_seg = np.where(cleaned_seg, cleaned_seg + max_label, cleaned_seg)
                # all_labels = np.unique(cleaned_seg)
                # skio.imsave(sav_dir + f"mask_x{x}_y{y}_z{z}.tif", arr=cleaned_seg)

                if x == 0:
                    list_of_labels = np.unique(cleaned_seg[:, -5:, :])
                    outside_labels = np.unique(cleaned_seg[:, :-5, :])
                    common_labels = np.intersect1d(outside_labels, list_of_labels)
                    remove_labels = np.setdiff1d(list_of_labels, common_labels)
                    # print(f"list_of_labels: {list_of_labels}")
                    # print(f"clean_seg_labels: {np.unique(cleaned_seg)}")
                    cleaned_seg = np.where( ~np.isin(cleaned_seg, remove_labels) , cleaned_seg, 0)
                    # print(f"cleaned_seg_labels_after: {np.unique(cleaned_seg)}")
                elif x + quad_size >= width:
                    list_of_labels = np.unique(cleaned_seg[:, :5, :])
                    outside_labels = np.unique(cleaned_seg[:, 5:, :])
                    common_labels = np.intersect1d(outside_labels, list_of_labels)
                    remove_labels = np.setdiff1d(list_of_labels, common_labels)
                    cleaned_seg = np.where( ~np.isin(cleaned_seg, remove_labels) , cleaned_seg, 0)
                else:
                    list_of_labels = np.unique(cleaned_seg[:, -5:, :])
                    outside_labels = np.unique(cleaned_seg[:, :-5, :])
                    common_labels = np.intersect1d(outside_labels, list_of_labels)
                    remove_labels = np.setdiff1d(list_of_labels, common_labels)
                    cleaned_seg = np.where( ~np.isin(cleaned_seg, remove_labels) , cleaned_seg, 0)
                    list_of_labels = np.unique(cleaned_seg[:, :5, :])
                    outside_labels = np.unique(cleaned_seg[:, 5:, :])
                    common_labels = np.intersect1d(outside_labels, list_of_labels)
                    remove_labels = np.setdiff1d(list_of_labels, common_labels)
                    cleaned_seg = np.where( ~np.isin(cleaned_seg, remove_labels) , cleaned_seg, 0)

                if y == 0:
                    list_of_labels = np.unique(cleaned_seg[:, :, -5:])
                    outside_labels = np.unique(cleaned_seg[:, :, :-5])
                    common_labels = np.intersect1d(outside_labels, list_of_labels)
                    remove_labels = np.setdiff1d(list_of_labels, common_labels)
                    cleaned_seg = np.where( ~np.isin(cleaned_seg, remove_labels) , cleaned_seg, 0)
                elif y + quad_size >= height:
                    list_of_labels = np.unique(cleaned_seg[:, :, :5])
                    outside_labels = np.unique(cleaned_seg[:, :, 5:])
                    common_labels = np.intersect1d(outside_labels, list_of_labels)
                    remove_labels = np.setdiff1d(list_of_labels, common_labels)
                    cleaned_seg = np.where( ~np.isin(cleaned_seg, remove_labels) , cleaned_seg, 0)
                else:
                    list_of_labels = np.unique(cleaned_seg[:, :, -5:])
                    outside_labels = np.unique(cleaned_seg[:, :, :-5])
                    common_labels = np.intersect1d(outside_labels, list_of_labels)
                    remove_labels = np.setdiff1d(list_of_labels, common_labels)
                    cleaned_seg = np.where( ~np.isin(cleaned_seg, remove_labels) , cleaned_seg, 0)
                    list_of_labels = np.unique(cleaned_seg[:, :, :5])
                    outside_labels = np.unique(cleaned_seg[:, :, 5:])
                    common_labels = np.intersect1d(outside_labels, list_of_labels)
                    remove_labels = np.setdiff1d(list_of_labels, common_labels)
                    cleaned_seg = np.where( ~np.isin(cleaned_seg, remove_labels) , cleaned_seg, 0)

                if z == 0:
                    list_of_labels = np.unique(cleaned_seg[-3:, :, :])
                    outside_labels = np.unique(cleaned_seg[:-3, :, :])
                    common_labels = np.intersect1d(outside_labels, list_of_labels)
                    remove_labels = np.setdiff1d(list_of_labels, common_labels)
                    cleaned_seg = np.where( ~np.isin(cleaned_seg, remove_labels) , cleaned_seg, 0)
                elif z + quad_depth >= depth_im:
                    list_of_labels = np.unique(cleaned_seg[:3, :, :])
                    outside_labels = np.unique(cleaned_seg[3:, :, :])
                    common_labels = np.intersect1d(outside_labels, list_of_labels)
                    remove_labels = np.setdiff1d(list_of_labels, common_labels)
                    cleaned_seg = np.where( ~np.isin(cleaned_seg, remove_labels) , cleaned_seg, 0)
                else:
                    list_of_labels = np.unique(cleaned_seg[-3:, :, :])
                    outside_labels = np.unique(cleaned_seg[:-3, :, :])
                    common_labels = np.intersect1d(outside_labels, list_of_labels)
                    remove_labels = np.setdiff1d(list_of_labels, common_labels)
                    cleaned_seg = np.where( ~np.isin(cleaned_seg, remove_labels) , cleaned_seg, 0)
                    list_of_labels = np.unique(cleaned_seg[:3, :, :])
                    outside_labels = np.unique(cleaned_seg[3:, :, :])
                    common_labels = np.intersect1d(outside_labels, list_of_labels)
                    remove_labels = np.setdiff1d(list_of_labels, common_labels)
                    cleaned_seg = np.where( ~np.isin(cleaned_seg, remove_labels) , cleaned_seg, 0)

                """ ADD IN THE NEW SEG??? or just let it overlap??? """
                # segmentation_labelled = np.copy(segmentation)

                incoming_region = segmentation[z:z+quad_depth, x:x+quad_size, y:y+quad_size]
                exist_labels = np.unique(incoming_region)[1:]

                for indx in exist_labels:
                    exist_mask_area = 0
                    new_mask_area = 0
                    if any(cleaned_seg[incoming_region == indx]) > 0:
                        exist_mask = np.zeros(segmentation.shape, dtype=np.int32)
                        exist_mask[segmentation == indx] = 1
                        new_label = np.unique(cleaned_seg[incoming_region == indx])
                        new_label = new_label[np.nonzero(new_label)] # find the non-zero overlapping pred label
                        # print(f"new_labels: {new_label}")
                        exist_mask_area = np.sum(exist_mask[exist_mask == 1])
                        for overlap in new_label:
                            # new_mask_large = np.zeros(segmentation.shape).astype(np.int32)
                            new_mask = np.zeros(cleaned_seg.shape, dtype=np.int32)
                            new_mask[cleaned_seg == overlap] = 1
                            new_mask_area = np.sum(new_mask[new_mask == 1])
                            if exist_mask_area < new_mask_area:
                                segmentation[segmentation == indx] = 0
                                break
                            else:
                                cleaned_seg[cleaned_seg == overlap] = 0
                segmentation[z:z+quad_depth, x:x+quad_size, y:y+quad_size] = cleaned_seg + segmentation[z:z+quad_depth,x:x+quad_size, y:y+quad_size]
    segmentation = measure.label(segmentation)
    segmentation = np.asarray(segmentation, dtype=np.uint32)
    # np.save(f"/cis/home/zchen163/my_documents/XTC_data/RSC03_roi1_buffer/medium_patches/p{input}.npy", segmentation)
    seg_memmap = np.memmap(f"/cis/home/zchen163/my_documents/XTC_data/RSC03_roi1_memmap/medium_patches/p{input}", dtype="uint32", mode='w+', shape=(depth_im, width, height))
    seg_memmap[:] = segmentation[:]
    seg_memmap.flush()
    del seg_memmap

def parallel_sparse(chunk, width=1024,height=1024,depth_im=121): #Image SIZE HERE GABY
    patches = np.array(np.arange(chunk*96,chunk*96+96))
    all_synapses = None
    for i in patches:
        file = open(f"/cis/home/gcoste1/MaskReg/MaskRegInference/InfPatchesStorage/p{i}.pkl", 'rb')
        synapse_locs = pickle.load(file)
        file.close()
        for locs in synapse_locs:
            temp_x, temp_y, temp_z = locs[0], locs[1], locs[2]
            # use temp_x * height * depth_im + temp_y * depth_im + temp_z to calculate the 1d coordinate
            loc_1d = temp_x * height * depth_im + temp_y * depth_im + temp_z
            loc_cols = np.zeros(loc_1d.shape)
            loc_data = np.ones(loc_1d.shape)
            temp_coo = coo_matrix((loc_data,(loc_1d,loc_cols)),shape=(width*height*depth_im,1))
            if all_synapses is None:
                all_synapses = temp_coo
            else:
                all_synapses = hstack([all_synapses,temp_coo],format="coo")
    return all_synapses

class InferenceDataset(Dataset):
    def __init__(self, input_im, stride, z_stride, quad_size, quad_depth):
        self.im = input_im
        depth_im, width, height = input_im.shape
        self.quad_size = quad_size
        self.quad_depth = quad_depth
        xs = []
        ys = []
        zs = []
        for x in range(0,width+3-quad_size,stride):
            if x + quad_size > width:
                difference = (x + quad_size) - width
                x = x - difference - 1
            xs.append(x)
        # print(f"x: {x}")
        for y in range(0,height+3-quad_size,stride):
            if y + quad_size > height:
                difference = (y + quad_size) - height
                y = y - difference - 1
            # print(f"y: {y}")
            ys.append(y)
        for z in range(0,depth_im+3-quad_depth,z_stride):
            if z + quad_depth > depth_im:
                difference = (z + quad_depth) - depth_im
                z = z - difference - 1
            zs.append(z)
        self.xs = xs
        self.ys = ys
        self.zs = zs

    def __len__(self):
        return len(self.xs) * len(self.ys) * len(self.zs)

    def __getitem__(self, idx):
        z_len = len(self.zs)
        y_len = len(self.ys)
        z = self.zs[idx%z_len]
        y = self.ys[int(np.floor(idx/z_len))%y_len]
        x = self.xs[int(np.floor(idx / (y_len * z_len)))]
        image = self.im[z:z+self.quad_depth, x:x+self.quad_size, y:y+self.quad_size]
        image = image.transpose((1,2,0))
        coord = [z,y,x]
        return image, coord

def check_overlap(indx, incoming_region, cleaned_seg):
    exist_mask_area = 0
    new_mask_area = 0
    seg_id = []
    new_id = []
    if any(cleaned_seg[incoming_region == indx]) > 0:
        exist_mask = np.zeros(cleaned_seg.shape, dtype=np.int32)
        exist_mask[incoming_region == indx] = 1
        new_label = np.unique(cleaned_seg[incoming_region == indx])
        new_label = new_label[np.nonzero(new_label)] # find the non-zero overlapping pred label
        # print(f"new_labels: {new_label}")
        exist_mask_area = np.sum(exist_mask[exist_mask==1])
        for overlap in new_label:
            new_mask = np.zeros(cleaned_seg.shape, dtype=np.int32)
            new_mask[cleaned_seg == overlap] = 1
            new_mask_area = np.sum(new_mask[new_mask==1])
            if exist_mask_area < new_mask_area:
                seg_id.append(indx)
            else:
                new_id.append(overlap)
    return [seg_id, new_id]

def resolve_overlaps(csr_masks,size_threshold=20):
    mask_sizes = csr_masks.sum(axis=0).A1
    sorted_indices = np.argsort(-mask_sizes)
    overlap_matrix = csr_masks.transpose().dot(csr_masks)
    overlap_matrix.setdiag(0)
    # full_mask = np.ones(len(coo_masks.data), dtype=bool)
    largest_in_overlaps = []
    smaller_overlaps = []
    # find connected components
    n_components, labels = connected_components(csgraph=overlap_matrix, directed=False)
    for component_label in tqdm(np.unique(labels), desc="Processing components", leave=False):
        component_indices = np.where(labels == component_label)[0]
        if len(component_indices) == 1:
            largest_in_overlaps.append(component_indices[0])
            continue
        largest_mask_index = component_indices[np.argmax(mask_sizes[component_indices])]
        for mask_index in component_indices:
            if mask_index != largest_mask_index:
                overlap_size = overlap_matrix[mask_index, largest_mask_index]
                if overlap_size < size_threshold:
                    smaller_overlaps.append(mask_index)
        largest_in_overlaps.append(largest_mask_index)
    return largest_in_overlaps, smaller_overlaps




if __name__=="__main__":
    class Args():
        def __init__(self):
            #self.dataset_name = "datasets/prostate"
            #self.dataset_name = "datasets/lidc"
            # self.dataset_name = "datasets/Rsc01_t1"
            # self.dataset_name = "datasets/Rsc01_t1_2"
            # self.dataset_name = "datasets/Rsc01_t1_group"
            # self.dataset_name = "datasets/Rsc01_t1_blacken"
            self.dataset_name = "datasets/Rsc03_shifted_all"
            # self.dataset_name = "datasets/Rsc01_t1_group_fliped_in_z"
            # self.dataset_name = "datasets/OL_data"
            # self.exp_dir = '/cis/home/zchen163/my_documents/regrcnn_96_96_32'
            # self.exp_dir = '/cis/home/zchen163/my_documents/regrcnn_96_96_32_100'
            # self.exp_dir = '/cis/home/zchen163/my_documents/regrcnn_96_96_32_shrink3'
            # self.exp_dir = '/cis/home/zchen163/my_documents/regrcnn_96_96_32_50_shrink3'
            # self.exp_dir = '/cis/home/zchen163/my_documents/regrcnn_96_96_32_50_conserv_shrink3'
            # self.exp_dir = '/cis/home/zchen163/my_documents/regrcnn_96_96_32_50_shrink3_fewer_anchors'
            # self.exp_dir = '/cis/home/zchen163/my_documents/regrcnn_96_96_32_50_shrink3_05_fewer_anchors'
            # self.exp_dir = '/cis/home/zchen163/my_documents/regrcnn_96_96_32_50_shrink3_05_fewer_anchors'
            # self.exp_dir = '/cis/home/zchen163/my_documents/regrcnn_96_96_32_50_nms02_edited_label'
            # self.exp_dir = '/cis/home/zchen163/my_documents/regrcnn_96_96_32_50_nms02_edited_label_GN'
            # self.exp_dir = '/cis/home/zchen163/my_documents/regrcnn_96_96_32_50_nms02_edited_label_GN_blacken'
            # self.exp_dir = '/cis/home/zchen163/my_documents/regrcnn_96_96_32_50_nms02_GN_shifted'
            self.exp_dir = '/cis/home/gcoste1/MaskReg/MaskRegInference/regrcnn_Rsc03_96_96_32_nms02_edited_GN_shifted4_all' #GABY TRAINING MODEL HERE
            # self.exp_dir = '/cis/home/zchen163/my_documents/regrcnn_96_96_32_50_nms02_edited_label_GN_shifted4_pyramid234'
            # self.exp_dir = '/cis/home/zchen163/my_documents/regrcnn_96_96_32_50_nms02_GN_shifted4'
            # self.exp_dir = '/cis/home/zchen163/my_documents/regrcnn_96_96_32_50_nms015_edited_label_GN'
            # self.exp_dir = '/cis/home/zchen163/my_documents/regrcnn_96_96_32_50_nms015_edited_label_IN'
            # self.exp_dir = '/cis/home/zchen163/my_documents/regrcnn_96_96_32_50_nms01_edited_label_GN'
            # self.exp_dir = '/cis/home/zchen163/my_documents/regrcnn_96_96_32_nms02_edited_label_GN'
            # self.exp_dir = '/cis/home/zchen163/my_documents/regrcnn_96_96_32_50_nms02_edited_label_GN_fliped_in_z'
            # self.exp_dir = '/cis/home/zchen163/my_documents/regrcnn_96_96_32_50_nms02_edited_label_GN_pyramid24'
            # self.exp_dir = '/cis/home/zchen163/my_documents/regrcnn_96_96_32_nms02_edited_label_IN'
            # self.exp_dir = '/cis/home/zchen163/my_documents/regrcnn_96_96_32_boundary_shrink3'
            # self.exp_dir = '/cis/home/zchen163/my_documents/regrcnn_96_96_32_100_boundary'
            # self.exp_dir = '/media/user/FantomHD/Lightsheet data/RegRCNN_maskrcnn_testing/'

            #self.exp_dir = '/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Training_blocks_RegRCNN/'
            # self.exp_dir = '/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Training_blocks_RegRCNN/15) 3D mrcnn rpn anchors 800_BEST_SO_FAR/'
            #self.exp_dir = '/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Training_blocks_RegRCNN/8) 2D_128x128x32 retina_unet/'


            # self.exp_dir = '/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Training_blocks_RegRCNN/22) 3D mrcnn post_nms_600_2000/'


            # self.exp_dir = '/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Training_blocks_RegRCNN/24) 3D mrcnn increased pre_nms_rois_and_other_stuff_BEST/'


            # self.exp_dir = '/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Training_blocks_RegRCNN/26) same as 24 but Resnet100/'




            self.server_env = False
    args = Args()

    config_file = utils.import_module('cf', os.path.join(args.exp_dir, "configs.py"))
    cf = config_file.Configs()
    cf.exp_dir = args.exp_dir
    cf.test_dir = cf.exp_dir

    #pid = '0811a'
    #cf.fold = find_pid_in_splits(pid)   ### TIGER -- super buggy for some reason...
    cf.fold = 0

    if cf.dim == 2:
        cf.merge_2D_to_3D_preds = True
        if cf.merge_2D_to_3D_preds:
            cf.dim==3

    else:
        cf.merge_2D_to_3D_preds = False

    cf.fold_dir = os.path.join(cf.exp_dir, 'fold_{}'.format(cf.fold))

    logger = utils.get_logger(cf.exp_dir)
    model = utils.import_module('model', os.path.join(cf.exp_dir, "model.py"))
    torch.backends.cudnn.benchmark = cf.dim == 3

    ### TIGER - missing currently ability to find best model


    """ TO LOAD OLD CHECKPOINT """
    # Read in file names
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    from natsort import natsort_keygen, ns
    natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order
    from os import listdir
    from os.path import isfile, join
    import glob, os
    onlyfiles_check = glob.glob(os.path.join(cf.fold_dir + '/','*_best_params.pth'))
    onlyfiles_check.sort(key = natsort_key1)

    # print(onlyfiles_check)
    """ Find last checkpoint """
    weight_path = onlyfiles_check[-1]   ### ONLY SOME CHECKPOINTS WORK FOR SOME REASON???
    print(f'weight_path: {weight_path}')

    """^^^ WHY DO ONLY SOME CHECKPOINTS WORK??? """

    # net = model.net(cf, logger).cuda(device)
    net1 = model.net(cf, logger)
    #pid = pids[0]
    #assert pid in pids

    # load already trained model weights
    rank = 0

    with torch.no_grad():
        pass
        net1.load_state_dict(torch.load(weight_path))
        net1.eval()
    # generate a batch from test set and show results
    # print(f"cuda device count: {torch.cuda.device_count()}")
    net = torch.nn.DataParallel(net1, device_ids=[0,1,2,4,5])
    net.to(f'cuda:{net.device_ids[0]}')
    net.eval()



    from natsort import natsort_keygen, ns
    natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order
    import glob, os

    #from csbdeep.internals import predict
    from tifffile import *
    import tkinter
    from tkinter import filedialog

    """ Select multiple folders for analysis AND creates new subfolder for results output """

    # list_folder = ["/cis/home/zchen163/my_documents/XTC_data/manual_sparse_test_masked/96_96_32_100"]
    # list_folder = ["/cis/home/zchen163/my_documents/XTC_data/Rsc01_t2_test_patch/"]
    # list_folder = ["/cis/home/zchen163/my_documents/XTC_data/RSC03_all/"]
    list_folder = ["/cis/home/gcoste1/MaskReg/RSC08/Test"] #GABY I THINK HERE IS THE FOLDER FOR
    # list_folder = ["/cis/home/zchen163/my_documents/XTC_data/Rsc03_roi1_t1_300_300_180/"]
    # list_folder = ["/cis/home/zchen163/my_documents/XTC_data/Rsc01_t2_test_patch_fliped_in_z/"]


    ###### start profiler
    profiler = cProfile.Profile()
    profiler.enable()
    """ Loop through all the folders and do the analysis!!!"""
    for input_path in list_folder:
        foldername = input_path.split('/')[-2]
        # sav_dir = input_path + '/' + foldername + '_output_96_96_32_100'
        # sav_dir = input_path + '/' + foldername + '_output_96_96_32_shrink3'
        # sav_dir = input_path + '/' + foldername + '_output_96_96_32_50_shrink3'
        # sav_dir = input_path + '/' + foldername + '_output_96_96_32_50_conserv_shrink3'
        # sav_dir = input_path + '/' + foldername + '_output_96_96_32_50_shrink3_fewer_anchors'
        # sav_dir = input_path + '/' + foldername + '_output_96_96_32_50_shrink3_05_fewer_anchors_full_existing_seg'
        # sav_dir = input_path + '/' + foldername + '_output_check_inference_with_masks'
        # sav_dir = input_path + '/' + foldername + '_output_check_inference_with_masks_thresh09'
        # sav_dir = input_path + '/' + foldername + '_output_check_inference_with_masks_thresh07'
        # sav_dir = input_path + '/' + foldername + '_output_check_inference_with_masks_iou'
        # sav_dir = input_path + '/' + foldername + '_output_check_inference_with_masks_iou_box_score'
        # sav_dir = input_path + '/' + foldername + '_output_96_96_32_50_nms02_edited_label_masks_iou_box_score'
        # sav_dir = input_path + '/' + foldername + '_output_96_96_32_50_nms02_edited_label_GN_masks_iou_box_score'
        sav_dir = input_path + '/' + foldername + '_output_check2'
        # sav_dir = input_path + '/' + foldername + '_output_96_96_32_50_nms02_edited_label_GN_shifted4_pyramid234_masks_iou'
        # sav_dir = input_path + '/' + foldername + '_output_96_96_32_50_nms015_edited_label_GN_masks_iou_box_score'
        # sav_dir = input_path + '/' + foldername + '_output_96_96_32_50_nms015_edited_label_IN_masks_iou_box_score'
        # sav_dir = input_path + '/' + foldername + '_output_96_96_32_50_nms01_edited_label_GN_masks_iou_box_score'
        # sav_dir = input_path + '/' + foldername + '_output_96_96_32_nms02_edited_label_GN_masks_iou_box_score'
        # sav_dir = input_path + '/' + foldername + '_output_96_96_32_nms02_edited_label_GN_masks_iou_box_score_sliding'
        # sav_dir = input_path + '/' + foldername + '_output_96_96_32_50_nms02_edited_label_GN_fliped_in_z_epoch600_masks_iou_box_score'
        # sav_dir = input_path + '/' + foldername + '_output_96_96_32_50_nms02_edited_label_GN_pyramid24_masks_iou_box_score'
        # sav_dir = input_path + '/' + foldername + '_output_96_96_32_nms02_edited_label_GN_small_stride_masks_iou_box_score'
        # sav_dir = input_path + '/' + foldername + '_output_96_96_32_nms02_edited_label_IN_masks_iou_box_score'

        """ For testing ILASTIK images """
        images = glob.glob(os.path.join(input_path,'*.tif'))    # can switch this to "*truth.tif" if there is no name for "input"
        images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
        examples = [dict(input=i,truth=i.replace('.tif','_truth.tif'), ilastik=i.replace('.tif','_single_Object Predictions_.tiff')) for i in images]



        try:
            # Create target Directory
            os.mkdir(sav_dir)
            print("Directory " , sav_dir ,  " Created ")
        except FileExistsError:
            print("Directory " , sav_dir ,  " already exists")

        sav_dir = sav_dir + '/'

        buffer_dir = '/cis/home/zchen163/my_documents/XTC_data/RSC03_roi1_buffer/'
        memmap_dir = '/cis/home/zchen163/my_documents/XTC_data/RSC03_roi1_memmap/'
        sparse_buffer_dir = "/cis/home/gcoste1/MaskReg/MaskRegInference/InfPatchesStorage/"

        # Required to initialize all
        for i in range(len(examples)):



            """ TRY INFERENCE WITH PATCH-BASED analysis from TORCHIO """
            with torch.set_grad_enabled(False):  # saves GPU RAM
                input_name = examples[i]['input']
                input_im = tiff.imread(input_name)
                # input_im = padding(input_im)
                # reference = skio.imread("/cis/home/zchen163/my_documents/XTC_data/Rsc03_06/RSc03_XTC_histmatch_ROI.tif")
                # reference = skio.imread("/cis/home/zchen163/my_documents/XTC_data/ilastik_seg/rsc01_reg_XTC_t1.tif")
                # input_im = match_histograms(input_im, reference)
                print(f"input_im.shape: {input_im.shape}")
                # print(input_im.dtype)
                # print(max(np.unique(input_im)))


                """ Analyze each block with offset in all directions """

                # Display the image
                #max_im = plot_max(input_im, ax=0)

                print('Starting inference on volume: ' + str(i) + ' of total: ' + str(len(examples)))
                #plot_max(input_im)


                overlap_percent = 0
                input_size = 96
                depth = 32
                num_truth_class = 2
                stride = 66 # 1024*1024*303
                # stride = 51  # 300*300*180
                z_stride = 15 # 1024*1024*303
                # z_stride = 17 # 300*300*180


                quad_size=input_size
                quad_depth=depth

                input_dataset = InferenceDataset(input_im, stride, z_stride, quad_size, quad_depth)
                test_dataloader = torch.utils.data.DataLoader(input_dataset, batch_size=1, shuffle=False)
                x_split = input_dataset.xs[int(np.floor(len(input_dataset.xs)/4))]
                z_split = input_dataset.zs[int(np.floor(len(input_dataset.zs)/3))]
                # np.floor(np.asarray(input_dataset.zs)/96)  # split z into 3 groups
                # np.floor(np.asarray(input_dataset.xs)/248) # split x into 4 groups
                # 48 groups of small patches
                skip_top=1


                thresh = 0.99
                cf.merge_3D_iou = thresh


                im_size = np.shape(input_im)
                width = im_size[1];  height = im_size[2]; depth_im = im_size[0]

                # segmentation_labelled = np.zeros([depth_im, width, height])
                # print(segmentation.shape)
                start_time = time.time()
                # num = 0
                # total_patches = np.empty((3,4,4),dtype=object)
                # shape_3d = total_patches.shape
                # for x in range(4):
                #     for y in range(4):
                #         for z in range(3):
                #             total_patches[z,x,y] = []

                segmentation = np.zeros([depth_im, width, height], dtype=np.int32)
                # all_synapses = []
                all_synapses = None
                count = 0
                for im, coord in tqdm(test_dataloader, desc="patches",leave=False):
                    # im = torch.permute(im, (0,2,3,1))  # torch 1.4.0 doesn't have permute
                    # print(f"coord:{coord}")
                    # print(f"im.shape before:{im.shape}")
                    im = im[:,None,:,:,:]
                    # print(f"im.shape:{im.shape}")
                    im = im.float().to(device)
                    _, _, _, detections, detection_masks = net.module.forward(im)
                    # print(f"detections:{detections[0:3]}")
                    # print(f"detection_masks:{detection_masks.size()}")
                    results_dict = net.module.get_results_modified(im.shape, detections, detection_masks, return_masks=True)
                    # print(f"coord:{coord}")
                    # z, y, x = coord
                    # z, y, x = int(z), int(y), int(x)

                    seg_im = results_dict['masks'][np.newaxis, np.newaxis, :]
                    synapse_locs = results_dict['sparse']   # a list of tuples contains x y z coordinates
                    # print(f"seg_im.shape:{seg_im.shape}")
                    # print(f"synapse_locs:{len(synapse_locs)}")
                    for batch_ix in range(len(synapse_locs)):
                        current_batch = synapse_locs[batch_ix]
                        z, y, x = coord[0][batch_ix], coord[1][batch_ix], coord[2][batch_ix]
                        z, y, x = int(z), int(y), int(x)
                        # print(f"z, y, x: {z, y, x}")
                        updated_locs = [(x+i, y+j, z+k) for i, j, k in current_batch]
                        file = open(sparse_buffer_dir+f'p{count}.pkl', 'wb')
                        pickle.dump(updated_locs, file)
                        file.close()
                        count += 1

                mins, secs = divmod((time.time() - start_time), 60)
                h, mins = divmod(mins, 60)
                t = "{:d}h:{:02d}m:{:02d}s".format(int(h), int(mins), int(secs))
                print("{} patch segmentation runtime: {}".format(os.path.split(__file__)[1], t))

                multi_start_time = time.time()

                num_processes = 4*4*3  # 48 patches
                if count < num_processes: #GABY CHANGED w/ ZHINING
                    num_processes = count
                patch_list = np.arange(count).tolist() #GABY CHANGED w/ ZHINING
                # pool = multiprocessing.Pool(processes=num_processes)
                # # outputs = pool.map(patch_inference, patch_list)
                # outputs = pool.map(parallel_sparse, patch_list)
                # pool.close()
                # pool.join()
                # all_synapses = hstack(outputs,format="coo")
                for i in patch_list:
                    file = open(f"/cis/home/gcoste1/MaskReg/MaskRegInference/InfPatchesStorage/p{i}.pkl", 'rb')
                    synapse_locs = pickle.load(file)
                    file.close()
                    for locs in synapse_locs:
                        temp_x, temp_y, temp_z = locs[0], locs[1], locs[2]
                        # use temp_x * height * depth_im + temp_y * depth_im + temp_z to calculate the 1d coordinate
                        loc_1d = temp_x * height * depth_im + temp_y * depth_im + temp_z
                        loc_cols = np.zeros(loc_1d.shape)
                        loc_data = np.ones(loc_1d.shape)
                        temp_coo = coo_matrix((loc_data,(loc_1d,loc_cols)),shape=(width*height*depth_im,1))
                        if all_synapses is None:
                            all_synapses = temp_coo
                        else:
                            all_synapses = hstack([all_synapses,temp_coo],format="coo")
                # sparse.save_npz("./sparse/all_synapses_lil.npz", all_synapses) ### lil is not efficient for storage

                mins, secs = divmod((time.time() - multi_start_time), 60)
                h, mins = divmod(mins, 60)
                t = "{:d}h:{:02d}m:{:02d}s".format(int(h), int(mins), int(secs))
                print("{} multiprocess runtime: {}".format(os.path.split(__file__)[1], t))

                    # if all_synapses.shape[1] >=200:
                    #     break
                # sparse.save_npz("./sparse/all_synapses_lil.npz", csr_synapses) ### lil is not efficient for storage

                csr_masks = all_synapses.tocsr()
                cleaned_masks = []
                largest_in_overlaps, smaller_overlaps = resolve_overlaps(csr_masks, size_threshold=20) # first round
                cleaned_masks.append(csr_masks[:,largest_in_overlaps].sum(axis=1))
                last_round = csr_masks
                while len(smaller_overlaps) > 2:
                    current_round = last_round[:,smaller_overlaps]
                    large, smaller_overlaps = resolve_overlaps(current_round, size_threshold=20)
                    cleaned_masks.append(current_round[:,large].sum(axis=1))
                    last_round = current_round
                # width, height, depth = 360, 360, 47 Gaby and Zhining commented out - already defined above
                merged_im = np.zeros((width, height, depth_im), dtype=np.uint32)
                for i in range(len(cleaned_masks)):
                    current_mask = cleaned_masks[i].A1
                    current_mask = current_mask.reshape((width, height, depth_im))
                    merged_im[current_mask>0] = i+1

                merged_im = np.transpose(merged_im,(2,0,1))
                labelled_im = measure.label(merged_im, connectivity=1)
                labelled_im = np.asarray(labelled_im, dtype=np.uint32)

            # segmentation = edit_labels(segmentation)

            filename = input_name.split('/')[-1].split('.')[0:-1]
            filename = '.'.join(filename)

            tiff.imwrite(sav_dir + filename + '_' + str(int(i)) +'_segmentation.tif', labelled_im)



            # ### if want unique labels:
            # from skimage import measure
            # labels = measure.label(segmentation)
            # labels = np.asarray(labels, dtype=np.uint32)
            # tiff.imwrite(sav_dir + filename + '_' + str(int(i)) +'_segmentation_LABELLED.tif', labels)

    profiler.disable() # end profiler

    # Create an output file
    output_file = "profiling_sparse_coo_parallel.txt"
    with open(output_file, 'w') as f:
        stats = io.StringIO()
        ps = pstats.Stats(profiler, stream=stats)
        ps.sort_stats('time')  # Sort statistics by time taken
        ps.print_stats()
        f.write(stats.getvalue())