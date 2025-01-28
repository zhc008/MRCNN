"""for presentations etc"""

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



""" Tiger function """
def plot_max(im, ax=0, plot=1):
    max_im = np.amax(im, axis=ax)
    if plot:
        plt.figure(); plt.imshow(max_im)
     
    return max_im



""" Tiger function """
def boxes_to_mask(cf, results_dict, thresh, unique_id=0):

    label_arr = np.copy(results_dict['seg_preds'],)
    new_labels = np.zeros(np.shape(results_dict['seg_preds']))
    class_labels = np.zeros(np.shape(results_dict['seg_preds']))
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
                box_arr[label_arr == 0] = 0
                
                new_labels[box_arr > 0] = box_id + 1
                
                class_labels[box_arr > 0] = box_row['box_pred_class_id']

                        
    label_arr = new_labels
    
    
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





if __name__=="__main__":
    class Args():
        def __init__(self):
            #self.dataset_name = "datasets/prostate"
            #self.dataset_name = "datasets/lidc"
            self.dataset_name = "datasets/Rsc01_t1"
            # self.dataset_name = "datasets/OL_data"
            
            #self.dataset_name = "datasets/Caspr_data"
            #self.exp_dir = "datasets/toy/experiments/mrcnnal2d_clkengal"  # detunet2d_di_bs16_ps512"
            #self.exp_dir = "/home/gregor/networkdrives/E132-Cluster-Projects/prostate/experiments/gs6071_retinau3d_cl_bs6"
            #self.exp_dir = "/home/gregor/networkdrives/E132-Cluster-Projects/prostate/experiments/gs6071_frcnn3d_cl_bs6"
            #self.exp_dir = "/home/gregor/networkdrives/E132-Cluster-Projects/prostate/experiments_t2/gs6071_mrcnn3d_cl_bs6_lessaug"
            #self.exp_dir = "/home/gregor/networkdrives/E132-Cluster-Projects/prostate/experiments/gs6071_detfpn3d_cl_bs6"
            #self.exp_dir = "/home/gregor/networkdrives/E132-Cluster-Projects/lidc_sa/experiments/ms12345_mrcnn3d_rgbin_bs8"
            #self.exp_dir = '/home/gregor/Documents/medicaldetectiontoolkit/datasets/lidc/experiments/ms12345_mrcnn3d_rg_bs8'
            #self.exp_dir = '/home/gregor/Documents/medicaldetectiontoolkit/datasets/lidc/experiments/ms12345_mrcnn3d_rgbin_bs8'
            self.exp_dir = '/cis/home/zchen163/my_documents/regrcnn_200epoch'
            # self.exp_dir = '/media/user/FantomHD/Lightsheet data/RegRCNN_maskrcnn_testing/'
            
            #self.exp_dir = '/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Training_blocks_RegRCNN/'
            # self.exp_dir = '/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Training_blocks_RegRCNN/15) 3D mrcnn rpn anchors 800_BEST_SO_FAR/'
            #self.exp_dir = '/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Training_blocks_RegRCNN/8) 2D_128x128x32 retina_unet/'
            
            
            # self.exp_dir = '/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Training_blocks_RegRCNN/22) 3D mrcnn post_nms_600_2000/'
            
            
            # self.exp_dir = '/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Training_blocks_RegRCNN/24) 3D mrcnn increased pre_nms_rois_and_other_stuff_BEST/'

            
            # self.exp_dir = '/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Training_blocks_RegRCNN/26) same as 24 but Resnet100/'
                        
            
                 

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
    
    
    
    cf.plot_dir = anal_dir ### TIGER ADDED FOR VAL_GEN
    val_gen = data_loader.get_train_generators(cf, logger, data_statistics=False)['val_sampling']
    batch_gen = data_loader.get_test_generator(cf, logger)
    #weight_paths = [os.path.join(cf.fold_dir, '{}_best_params.pth'.format(rank)) for rank in
    #                test_predictor.epoch_ranking]
    #weight_path = weight_paths[rank]
    
    ### TIGER - missing currently ability to find best model
    

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
    onlyfiles_check.sort(key = natsort_key1)
    
    # print(onlyfiles_check)
    """ Find last checkpoint """       
    weight_path = onlyfiles_check[-1]   ### ONLY SOME CHECKPOINTS WORK FOR SOME REASON???
    
    """^^^ WHY DO ONLY SOME CHECKPOINTS WORK??? """
    
    #split = last_file.split('check_')[-1]
    #num_check = split.split('.')
    #checkpoint = num_check[0]
    #checkpoint = 'check_' + checkpoint
    #num_check = int(num_check[0])
    
    #check = torch.load(s_path + checkpoint, map_location=device)
    
    
    #unet = check['model_type']
    #unet.load_state_dict(check['model_state_dict'])
    #unet.eval()
    #unet.training # check if mode set correctly
    #unet.to(device)
    

    net = model.net(cf, logger).cuda(device)
    
    #weight_path = os.path.join(cf.fold_dir, '77_best_params.pth') 
    
    
    #weight_path = os.path.join(cf.fold_dir, '251_best_params.pth') 
    
    #weight_path = os.path.join(cf.fold_dir, '68_best_params.pth') 
    
    
    
    try:
        pids = batch_gen["test"].dataset_pids
    except:
        pids = batch_gen["test"].generator.dataset_pids
    print("pids in test set: ",  pids)
    #pid = pids[0]
    #assert pid in pids

    # load already trained model weights
    rank = 0
    
    with torch.no_grad():
        pass
        net.load_state_dict(torch.load(weight_path))
        net.eval()
    # generate a batch from test set and show results
    if not os.path.isdir(anal_dir):
        os.mkdir(anal_dir)

    
    #plot_train_forward(val_gen)
    #plot_forward('906')
    
    num_to_plot = 50
    plot_boxes = 0
    
    thresh_2D_to_3D_boxes = 0.5
    
    
    from natsort import natsort_keygen, ns
    natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order
    import glob, os
    
    #from csbdeep.internals import predict
    from tifffile import *
    import tkinter
    from tkinter import filedialog
        
    """ Select multiple folders for analysis AND creates new subfolder for results output """
    # root = tkinter.Tk()
    # # get input folders
    # another_folder = 'y';
    # list_folder = []
    # input_path = "./"
    
    # initial_dir = '/media/user/storage/Data/'
    # while(another_folder == 'y'):
    #     input_path = filedialog.askdirectory(parent=root, initialdir= initial_dir,
    #                                         title='Please select input directory')
    #     input_path = input_path + '/'
        
    #     print('Do you want to select another folder? (y/n)')
    #     another_folder = input();   # currently hangs forever
    #     #another_folder = 'y';
    
    #     list_folder.append(input_path)
    #     initial_dir = input_path    
        

    list_folder = ["/cis/home/zchen163/my_documents/XTC_data/ilastik_rsc01_t1/"]



    """ Loop through all the folders and do the analysis!!!"""
    for input_path in list_folder:
        foldername = input_path.split('/')[-2]
        sav_dir = input_path + '/' + foldername + '_output_200epoch'
    
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
        
        # Required to initialize all
        batch_size = 1;
        
        batch_x = []; batch_y = [];
        weights = [];
        
        plot_jaccard = [];
        
        output_stack = [];
        output_stack_masked = [];
        all_PPV = [];
        input_im_stack = [];
        for i in range(len(examples)):
             
        
            
             """ TRY INFERENCE WITH PATCH-BASED analysis from TORCHIO """
             with torch.set_grad_enabled(False):  # saves GPU RAM            
                input_name = examples[i]['input']            
                input_im = tiff.imread(input_name)
                
                

                
       
                """ Analyze each block with offset in all directions """
                
                # Display the image
                #max_im = plot_max(input_im, ax=0)
                
                print('Starting inference on volume: ' + str(i) + ' of total: ' + str(len(examples)))
                #plot_max(input_im)
                
                # segmentation = UNet_inference_by_subparts_PYTORCH(unet, device, input_im, overlap_percent, quad_size=input_size, quad_depth=depth,
                #                                           mean_arr=mean_arr, std_arr=std_arr, num_truth_class=num_truth_class,
                #                                           skip_top=1)        
            
                overlap_percent = 0
                input_size = 96
                depth = 48
                num_truth_class = 2
                stride = 58
                z_stride = 31
                
                
                quad_size=input_size
                quad_depth=depth
                skip_top=1
                
                
                thresh = 0.99
                cf.merge_3D_iou = thresh
                
                
                im_size = np.shape(input_im);
                width = im_size[1];  height = im_size[2]; depth_im = im_size[0];
                    
                segmentation = np.zeros([depth_im, width, height])
                total_blocks = 0;
                all_xyz = []                                               
                 
                    
                for x in range(0, width + quad_size, stride):
                    if x + quad_size > width:
                           difference = (x + quad_size) - width
                           x = x - difference
                                
                    for y in range(0, height + quad_size, stride):
                           
                            if y + quad_size > height:
                                difference = (y + quad_size) - height
                                y = y - difference
                           
                            for z in range(0, depth_im + quad_depth, z_stride):
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
                               
                                quad_intensity = input_im[z:z + quad_depth,  x:x + quad_size, y:y + quad_size];  
                               
                          
                               
                                quad_intensity = np.moveaxis(quad_intensity, 0, -1)
                                quad_intensity = np.expand_dims(quad_intensity, axis=0)
                                quad_intensity = np.expand_dims(quad_intensity, axis=0)
                                quad_intensity = np.asarray(quad_intensity, dtype=np.float16)
                               
                                batch = {'data':quad_intensity, 'seg': np.zeros([1, 1, input_size, input_size, depth]), 
                                        'class_targets': np.asarray([]), 'bb_targets': np.asarray([]), 
                                        'roi_masks': np.zeros([1, 1, 1, input_size, input_size, depth]),
                                        'patient_bb_target': np.asarray([]), 'original_img_shape': quad_intensity.shape,
                                        'patient_class_targets': np.asarray([]), 'pid': ['0']}
                               
                               
                                # class_targets': array([], shape=(1, 0), dtype=float64),
                                # 'bb_target': array([], shape=(1, 0), dtype=float64),
                                # 'roi_masks': array([[[[[[0, 0, 0, ..., 0, 0, 0],
                                #            [0, 0, 0, ..., 0, 0, 0],

                                #batch.keys()
                                #dict_keys(['data', 'seg', 'class_targets', 'bb_target', 'roi_masks', 'patient_bb_target', 'original_img_shape', 'patient_class_targets', 'pid'])
                                # batch['data'].shape
                                # (1, 3, 128, 128, 32)
                                # ]], dtype=uint32),
                                #  'patient_bb_target': array([], shape=(1, 0), dtype=float64),
                                #  'original_img_shape': (1, 3, 128, 128, 32),
                                #  'patient_class_targets': array([], shape=(1, 0), dtype=float64),
                                #  'pid': array(['0'], dtype='<U1')}



                                results_dict = net.test_forward(batch) #seg preds are only seg_logits! need to take argmax.
                                
                                
                                
                                if 'seg_preds' in results_dict.keys():
                                        results_dict['seg_preds'] = np.argmax(results_dict['seg_preds'], axis=1)[:,np.newaxis]
                            
                                #out_file = os.path.join(anal_dir, "straight_inference_fold_{}_pid_{}".format(str(cf.fold), pid))
                                    
                                    
                                print(results_dict['seg_preds'].shape)
                                print(batch['data'].shape)
                                
                               
                    
                        
                                import matplotlib
                                matplotlib.use('Qt5Agg')    
                                    ### TIGER - SAVE AS TIFF
                                if cf.dim == 2:
                                        #input_im = np.expand_dims(batch['data'], axis=0)
                                        truth_im = np.expand_dims(batch['seg'], axis=0)
                                        seg_im = np.expand_dims(results_dict['seg_preds'], axis=0)
                                        
                                        
                                        """ roi_mask means we dont need bounding boxes!!! 
                                        
                                                - if there are NO objects in this image, then will have a weird shape, so need to parse it by len()
                                                - otherwise, it is a list of lists (1 list for each slice, which contains arrays for every object on that slice)
                                        """
                                        if len(np.unique(seg_im)) == 1:
                                            continue   ### no objects found (only 0 - background)
                                            
                                            
                                        elif cf.merge_2D_to_3D_preds:
                                            """ NEED MORE WORK IF WANT TO CONVERT 2D to 3D"""
                                            
                                            print('merge 2D to 3D with iou thresh of: ' + str(cf.merge_3D_iou))
                                            
                                            import predictor as pred
                                            results_2to3D = {}
                                            results_2to3D['2D_boxes'] = results_dict['boxes']
                                            merge_dims_inputs = [results_dict['boxes'], 'dummy_pid', cf.class_dict, cf.merge_3D_iou]
                                            results_2to3D['boxes'] = pred.apply_2d_3d_merging_to_patient(merge_dims_inputs)[0]
                                            
                        
                                            label_arr, class_labels = boxes_to_mask(cf, results_dict=results_2to3D, thresh=cf.merge_3D_iou)
                                            
                                            
                                        class_labels = np.asarray(class_labels, dtype=np.uint32)
                                            
                                        class_labels = np.expand_dims(class_labels, axis=0)


                                elif cf.dim == 3:
                                        
                                        
                                        #input_im = np.moveaxis(batch['data'], -1, 1) 
                                        truth_im = np.moveaxis(batch['seg'], -1, 1) 
                                        seg_im = np.moveaxis(results_dict['seg_preds'], -1, 1) 
                                        
                                        
                                        """ roi_mask means we dont need bounding boxes!!! 
                                        
                                                - if there are NO objects in this image, then will have a weird shape, so need to parse it by len()
                                                - otherwise, it is a list of lists (1 list for each slice, which contains arrays for every object on that slice)
                                        """
                                        
                                        if len(np.unique(seg_im)) == 1:
                                            continue   ### no objects found (only 0 - background) 
                                        else:
                                            
                                            label_arr, class_labels = boxes_to_mask(cf, results_dict=results_dict, thresh=cf.merge_3D_iou)
                                            
                                            if len(np.unique(label_arr)) == 1:
                                                continue   ### no objects found (only 0 - background)      
                                        
                                        # class_labels = np.asarray(class_labels, dtype=np.uint32)   
                                        # class_labels = np.moveaxis(class_labels, -1, 1)         
                                        
                                        
                                        ### if want colorful mask split up by boxes
                                        label_arr = np.asarray(label_arr, dtype=np.uint32)   
                                        label_arr = np.moveaxis(label_arr, -1, 1)                                      
                                        

                    


                                cleaned_seg = np.asarray(label_arr, dtype=np.uint32)
                                cleaned_seg = cleaned_seg[0, :, 0, :, :]
                                
                                
                                
                                
                                """ ADD IN THE NEW SEG??? or just let it overlap??? """                     
                                
                                ### this simply uses the new seg
                                #    segmentation[z:z + quad_depth, x:x + quad_size, y:y + quad_size] = cleaned_seg       
                                
                                
                                ### try use max with previous seg for the overlapping part
                                segmentation[z:z+quad_depth-z_stride, x:x+quad_size-stride, y:y+quad_size-stride] = np.maximum(
                                    cleaned_seg[:quad_depth-z_stride, :quad_size-stride, :quad_size-stride], 
                                    segmentation[z:z+quad_depth-z_stride, x:x+quad_size-stride, y:y+quad_size-stride])
                                
                                segmentation[z+quad_depth-z_stride:z+quad_depth, x+quad_size-stride:x+quad_size, y+quad_size-stride:y+quad_size] = cleaned_seg[
                                                                                                        quad_depth-z_stride:, quad_size-stride:, quad_size-stride:]
                                
                                

                                ### this is adding, it is bad
                                #segmentation[z:z + quad_depth, x:x + quad_size, y:y + quad_size] = cleaned_seg + segmentation[z:z + quad_depth, x:x + quad_size, y:y + quad_size]
                        
                                total_blocks += 1
                                
                                

                                
                                
                                

                    #segmentation[segmentation > 0] = 255
                    filename = input_name.split('/')[-1].split('.')[0:-1]
                    filename = '.'.join(filename)
                    
                    ### if operating system is Windows, must also remove \\ slash
                    #if os_windows:
                    #     filename = filename.split('\\')[-1]
                        
                        
                    segmentation = np.asarray(segmentation, np.uint32)
                    tiff.imwrite(sav_dir + filename + '_' + str(int(i)) +'_segmentation.tif', segmentation)
                    segmentation[segmentation > 0] = 1
                    
                    #input_im = np.asarray(input_im, np.uint32)
                    input_im = np.expand_dims(input_im, axis=0)
                    input_im = np.expand_dims(input_im, axis=2)
                    tiff.imwrite(sav_dir + filename + '_' + str(int(i)) +'_input_im.tif', input_im,
                                        imagej=True, #resolution=(1/XY_res, 1/XY_res),
                                        metadata={'spacing':1, 'unit': 'um', 'axes': 'TZCYX'})  
                            

            
            
            
            
                                
                    ### if want unique labels:
                    from skimage import measure
                    labels = measure.label(segmentation)
                    labels = np.asarray(labels, dtype=np.uint32)
                    tiff.imwrite(sav_dir + filename + '_' + str(int(i)) +'_segmentation_LABELLED.tif', labels)
                

            
            
            
            
            
            
            
            
            
            
            
            
            
            # for id_f in range(0, len(pids)):
                
            #     pid = pids[id_f]
            
            #     with torch.no_grad():
            #         batch = batch_gen['test'].generate_train_batch(pid=pid)
            #         results_dict = net.test_forward(batch) #seg preds are only seg_logits! need to take argmax.
            
            #         if 'seg_preds' in results_dict.keys():
            #             results_dict['seg_preds'] = np.argmax(results_dict['seg_preds'], axis=1)[:,np.newaxis]
            
            #         out_file = os.path.join(anal_dir, "straight_inference_fold_{}_pid_{}".format(str(cf.fold), pid))
                    
                    
            #         print(results_dict['seg_preds'].shape)
            #         print(batch['data'].shape)
                    
            #         zzz
        
        
            #         import matplotlib
            #         matplotlib.use('Qt5Agg')    
            #         ### TIGER - SAVE AS TIFF
            #         if cf.dim == 2:
            #             input_im = np.expand_dims(batch['data'], axis=0)
            #             truth_im = np.expand_dims(batch['seg'], axis=0)
            #             seg_im = np.expand_dims(results_dict['seg_preds'], axis=0)
                        
        
            #             tiff.imwrite(out_file + '_input_im.tif', np.asarray(input_im, dtype=np.uint32),
            #                           imagej=True,   metadata={'spacing': 1, 'unit': 'um', 'axes': 'TZCYX'})
                        
            #             tiff.imwrite(out_file + '_TRUTH.tif', np.asarray(truth_im, dtype=np.uint32),
            #                           imagej=True,   metadata={'spacing': 1, 'unit': 'um', 'axes': 'TZCYX'})
                                    
            #             tiff.imwrite(out_file + '_seg.tif', np.asarray(seg_im, dtype=np.uint32),
            #                           imagej=True,   metadata={'spacing': 1, 'unit': 'um', 'axes': 'TZCYX'})
        
        
                        
            #             """ roi_mask means we dont need bounding boxes!!! 
                        
            #                     - if there are NO objects in this image, then will have a weird shape, so need to parse it by len()
            #                     - otherwise, it is a list of lists (1 list for each slice, which contains arrays for every object on that slice)
            #             """
            #             if len(np.unique(seg_im)) == 1:
            #                 continue   ### no objects found (only 0 - background)
                            
                            
            #             elif cf.merge_2D_to_3D_preds:
            #                 """ NEED MORE WORK IF WANT TO CONVERT 2D to 3D"""
                            
            #                 print('merge 2D to 3D with iou thresh of: ' + str(cf.merge_3D_iou))
                            
            #                 import predictor as pred
            #                 results_2to3D = {}
            #                 results_2to3D['2D_boxes'] = results_dict['boxes']
            #                 merge_dims_inputs = [results_dict['boxes'], 'dummy_pid', cf.class_dict, cf.merge_3D_iou]
            #                 results_2to3D['boxes'] = pred.apply_2d_3d_merging_to_patient(merge_dims_inputs)[0]
                            
        
            #                 label_arr = boxes_to_mask(cf, results_dict=results_2to3D, thresh=cf.merge_3D_iou)
                            
                                                    
                        
            #             label_arr = np.asarray(label_arr, dtype=np.uint32)
                            
            #             label_arr = np.expand_dims(label_arr, axis=0)
                
            #             tiff.imwrite(out_file + '_roi_masks_LABEL.tif', np.asarray(label_arr, dtype=np.uint32),
            #                           imagej=True,   metadata={'spacing': 1, 'unit': 'um', 'axes': 'TZCYX'})        
                        
            #             ### This below hangs
                        
            #             # plg.view_batch(cf, batch, res_dict=results_dict, show_info=False, legend=True, show_gt_labels=True,
            #             #                           out_file=out_file, sample_picks=slices, has_colorchannels=False)
            #             if plot_boxes:
            #                 print('Plotting boxes png')
            #                 utils.split_off_process(plg.view_batch, cf, batch, results_dict, has_colorchannels=cf.has_colorchannels,
            #                                         show_gt_labels=True, get_time="val-example plot",
            #                                         out_file=os.path.join(cf.plot_dir, 'batch_SINGLE_PID_{}.png'.format(pid)))
                            
                    
                
            #         ### TIGER - SAVE AS TIFF
            #         elif cf.dim == 3:
                        
                        
            #             input_im = np.moveaxis(batch['data'], -1, 1) 
            #             truth_im = np.moveaxis(batch['seg'], -1, 1) 
            #             seg_im = np.moveaxis(results_dict['seg_preds'], -1, 1) 
                        
        
            #             tiff.imwrite(out_file + '_input_im.tif', np.asarray(input_im, dtype=np.uint32),
            #                           imagej=True,   metadata={'spacing': 1, 'unit': 'um', 'axes': 'TZCYX'})
                        
            #             tiff.imwrite(out_file + '_TRUTH.tif', np.asarray(truth_im, dtype=np.uint32),
            #                           imagej=True,   metadata={'spacing': 1, 'unit': 'um', 'axes': 'TZCYX'})
                                    
            #             tiff.imwrite(out_file + '_seg.tif', np.asarray(seg_im, dtype=np.uint32),
            #                           imagej=True,   metadata={'spacing': 1, 'unit': 'um', 'axes': 'TZCYX'})
        
        
                        
            #             """ roi_mask means we dont need bounding boxes!!! 
                        
            #                     - if there are NO objects in this image, then will have a weird shape, so need to parse it by len()
            #                     - otherwise, it is a list of lists (1 list for each slice, which contains arrays for every object on that slice)
            #             """
                        
            #             if len(np.unique(seg_im)) == 1:
            #                 continue   ### no objects found (only 0 - background)
                                    
            #             else:
                            
            #                 label_arr = boxes_to_mask(cf, results_dict=results_dict, thresh=cf.merge_3D_iou)
                            
            #                 if len(np.unique(label_arr)) == 1:
            #                     continue   ### no objects found (only 0 - background)
                                        
                            
                        
            #             label_arr = np.asarray(label_arr, dtype=np.uint32)
                            
            #             label_arr = np.moveaxis(label_arr, -1, 1) 
                
            #             tiff.imwrite(out_file + '_roi_masks_LABEL.tif', np.asarray(label_arr, dtype=np.uint32),
            #                           imagej=True,   metadata={'spacing': 1, 'unit': 'um', 'axes': 'TZCYX'})        
                        
            #             ### This below hangs
                        
            #             # plg.view_batch(cf, batch, res_dict=results_dict, show_info=False, legend=True, show_gt_labels=True,
            #             #                           out_file=out_file, sample_picks=slices, has_colorchannels=False)
            #             if plot_boxes:
            #                 print('Plotting boxes png')
            #                 utils.split_off_process(plg.view_batch, cf, batch, results_dict, has_colorchannels=cf.has_colorchannels,
            #                                         show_gt_labels=True, get_time="val-example plot",
            #                                         out_file=os.path.join(cf.plot_dir, 'batch_SINGLE_PID_{}.png'.format(pid)))
                            
                    
        


