"""for presentations etc"""

from cProfile import label
from inspect import cleandoc
import plotting as plg

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
from skimage.measure import regionprops


def padding(im, pad_size=8, axis=0):
    """
    Pads a 3D image array along a given axis with zeros.

    Parameters:
        im (ndarray): Input 3D image (shape: D x H x W).
        pad_size (int): Amount of zero padding on each side.
        axis (int): Axis to pad (default is 0, i.e., depth).

    Returns:
        ndarray: Zero-padded image.
    """
    padded_size = im.shape[axis] + 2 * pad_size
    output = np.zeros((padded_size, im.shape[1], im.shape[2])).astype(im.dtype)
    output[pad_size:-pad_size,:,:] = im
    return output


def reindex_labels(im):
    """
    Reindexes labels to be consecutive integers starting from 1 to the number of labels.

    Parameters:
        im (ndarray): Label mask.
    """
    labels = np.unique(im)
    for i in range(labels.shape[0]):
        im[im == labels[i]] = i


if __name__=="__main__":
    class Args():
        def __init__(self):
            self.dataset_name = "../datasets/Rsc03_shifted_all"
            self.exp_dir = "../../regrcnn_Rsc03_96_96_32_nms02_edited_GN_shifted4_all"   

            self.server_env = False
    args = Args()


    data_loader = utils.import_module('dl', os.path.join(args.dataset_name, "data_loader.py"))

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
    anal_dir = os.path.join(cf.exp_dir, "inference_analysis")

    logger = utils.get_logger(cf.exp_dir)
    model = utils.import_module('model', os.path.join(cf.exp_dir, "model.py"))
    torch.backends.cudnn.benchmark = cf.dim == 3
    
    test_predictor = Predictor(cf, None, logger, mode='test')
    test_evaluator = Evaluator(cf, logger, mode='test')
    

    """ TO LOAD OLD CHECKPOINT """
    # Read in file names
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    from natsort import natsort_keygen, ns
    natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order
    from os import listdir
    from os.path import isfile, join
    import glob, os
    onlyfiles_check = glob.glob(os.path.join(cf.fold_dir + '/','*_best_params.pth'))
    # onlyfiles_check = glob.glob(os.path.join(cf.fold_dir + '/','last_state.pth'))
    onlyfiles_check.sort(key = natsort_key1)
    
    # print(onlyfiles_check)
    """ Find last checkpoint """       
    weight_path = onlyfiles_check[-1]   
    print(f'weight_path: {weight_path}')

    net = model.net(cf, logger).cuda(device)
    

    # load already trained model weights
    rank = 0
    
    with torch.no_grad():
        pass
        net.load_state_dict(torch.load(weight_path))
        net.eval()
    # generate a batch from test set and show results
    if not os.path.isdir(anal_dir):
        os.mkdir(anal_dir)
    
    num_to_plot = 50
    plot_boxes = 0
    
    thresh_2D_to_3D_boxes = 0.5
    
    #from csbdeep.internals import predict
    from tifffile import *
    import tkinter
    from tkinter import filedialog
        
    """ Select multiple folders for analysis AND creates new subfolder for results output """ 
    list_folder = ["../../RSC02_roi3_GC"]

    """ Loop through all the folders and do the analysis!!!"""
    for input_path in list_folder:
        foldername = input_path.split('/')[-2]

        sav_dir = input_path + '/' + foldername + '_output'


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
        
        for i in range(len(examples)):
                 
            """ TRY INFERENCE WITH PATCH-BASED analysis from TORCHIO """
            with torch.set_grad_enabled(False):  # saves GPU RAM            
                input_name = examples[i]['input']            
                input_im = tiff.imread(input_name)
                
       
                """ Analyze each block with offset in all directions """
                
                
                print('Starting inference on volume: ' + str(i) + ' of total: ' + str(len(examples)))
       
            
                overlap_percent = 0
                input_size = 96
                depth = 32
                num_truth_class = 2
                stride = 68
                # stride = 1
                z_stride = 32
                
                
                quad_size=input_size
                quad_depth=depth
                skip_top=1
                
                
                thresh = 0.99
                cf.merge_3D_iou = thresh
                
                
                im_size = np.shape(input_im)
                width = im_size[1];  height = im_size[2]; depth_im = im_size[0]
                    
                segmentation = np.zeros([depth_im, width, height])
                segmentation_labelled = np.zeros([depth_im, width, height])
                # print(segmentation.shape)
                total_blocks = 0
                all_xyz = [] 
                num_masks = 0                                              
                # k = 0
                # add a counter for each patch's seg_pred
                patch_counter = 1
                for x in range(1):
                    if x + quad_size > width:
                        difference = (x + quad_size) - width
                        x = x - difference
                            
                    for y in range(1):
                        
                        if y + quad_size > height:
                            difference = (y + quad_size) - height
                            y = y - difference
                        
                        for z in range(1):
                            #batch_x = []; batch_y = [];
                    
                            if z + quad_depth > depth_im:
                                difference = (z + quad_depth) - depth_im
                                z = z - difference
                            
                                
                            """ Check if repeated """
                            skip = 0
                            for coord in all_xyz:
                                if coord == [x,y,z]:
                                        skip = 1
                                        break                      
                            if skip:  continue
                                
                            all_xyz.append([x, y, z])


                            # 96 by 96 by 32 training
                            quad_intensity = input_im[z:z + quad_depth,  x:x + quad_size, y:y + quad_size];  
                            
                            quad_intensity = np.moveaxis(quad_intensity, 0, -1)
                            quad_intensity = np.expand_dims(quad_intensity, axis=0)
                            quad_intensity = np.expand_dims(quad_intensity, axis=0)
                            quad_intensity = np.asarray(quad_intensity, dtype=np.float16)
                            # print(max(np.unique(quad_intensity)))
                            # print(quad_intensity.shape)
                            
                            batch = {'data':quad_intensity, 'seg': np.zeros([1, 1, input_size, input_size, depth]), 
                                    'class_targets': np.asarray([]), 'bb_targets': np.asarray([]), 
                                    'roi_masks': np.zeros([1, 1, 1, input_size, input_size, depth]),
                                    'patient_bb_target': np.asarray([]), 'original_img_shape': quad_intensity.shape,
                                    'patient_class_targets': np.asarray([]), 'pid': ['0']}

                            results_dict = net.test_forward_modified(batch) #seg preds are only seg_logits! need to take argmax.
                       

                            seg_im = results_dict['masks'][np.newaxis, np.newaxis, :]
                                        
                
                
                            import matplotlib
                            matplotlib.use('Qt5Agg')    
                            ### TIGER - SAVE AS TIFF
                            if cf.dim == 2:
                                #input_im = np.expand_dims(batch['data'], axis=0)
                                truth_im = np.expand_dims(batch['seg'], axis=0)
                                seg_im = np.expand_dims(results_dict['seg_preds'], axis=0)
                            
                                if len(np.unique(seg_im)) == 1:
                                    continue   ### no objects found (only 0 - background)
                            
                                class_labels = np.asarray(class_labels, dtype=np.uint32)
                                    
                                class_labels = np.expand_dims(class_labels, axis=0)


                            elif cf.dim == 3:

                                truth_im = np.moveaxis(batch['seg'], -1, 1) 

                                
                                patch_counter += 1

                                if len(np.unique(seg_im)) == 1:
                                    continue   ### no objects found (only 0 - background) 
                                else:
                                    
                                    # label_arr, class_labels = boxes_to_mask(cf, results_dict=results_dict, thresh=cf.merge_3D_iou, input_im=quad_intensity)

                                    label_arr = seg_im.copy()
                                    label_arr[label_arr > 0] + num_masks
                                    num_masks += np.unique(seg_im).shape[0] - 1 # add current number of segmentations to the label to make each label different
                                    
                                    if len(np.unique(label_arr)) == 1:
                                        continue   ### no objects found (only 0 - background)            
                                
                                
                                ### if want colorful mask split up by boxes
                                label_arr = np.asarray(label_arr, dtype=np.uint32)   
                                label_arr = np.moveaxis(label_arr, -1, 1)                                      
                                

                


                            cleaned_seg = np.asarray(label_arr, dtype=np.uint32)
                            # print(cleaned_seg.shape)
                            cleaned_seg = cleaned_seg[0, :, 0, :, :]

                            segmentation = cleaned_seg
                            
            #segmentation[segmentation > 0] = 255
            filename = input_name.split('/')[-1].split('.')[0:-1]
            filename = '.'.join(filename)
     
            segmentation = np.asarray(segmentation, np.uint32)
            # tiff.imwrite(sav_dir + filename + '_' + str(int(i)) +'_segmentation.tif', segmentation)

                            
            ### if want unique labels:
            from skimage import measure
            labels = measure.label(segmentation)
            labels = np.asarray(labels, dtype=np.uint32)
            tiff.imwrite(sav_dir + filename + '_' + str(int(i)) +'_segmentation_LABELLED.tif', labels)
            # tiff.imwrite(sav_dir + 'crop_' + str(int(i)) +'.tif', labels)