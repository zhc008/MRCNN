"""for presentations etc"""

from cProfile import label
from inspect import cleandoc

import sys
import os
import pickle

import numpy as np
import pandas as pd
import torch

import utils.exp_utils as utils
import utils.model_utils as mutils

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
from scipy.sparse import coo_matrix, hstack, diags
from scipy import sparse
from scipy.sparse.csgraph import connected_components


def edit_labels(waterim):
    """
    Cleans and relabels 3D segmentation masks by:
    - Removing small components (<=3 pixels in area).
    - Removing linear segments (eccentricity == 1).
    - Removing components that span fewer than 3 z-planes.
    - Relabeling the mask with connectivity=1.

    Parameters:
        waterim (ndarray): 3D segmentation mask.

    Returns:
        ndarray: Cleaned and relabeled 3D masks.
    """
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

        #Remove small components and those that span less than 3 z-planes
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

def parallel_sparse(chunk, width=1024,height=1024,depth_im=303, chunk_size=76):
    """
    Loads patch-based sparse masks and assembles them into a large sparse matrix for a chunk.

    Parameters:
        chunk (int): Chunk index.
        width, height, depth_im (int): Dimensions of full volume.
        chunk_size (int): Number of patches in each chunk.

    Returns:
        coo_matrix: Sparse matrix containing all synapse voxels from the chunk.
    """
    patches = np.array(np.arange(chunk*chunk_size,chunk*chunk_size+chunk_size))
    all_synapses = None
    for i in patches:
        try:
            file = open(f"/cis/home/gcoste1/MaskReg/MaskRegInference/InfPatchesStorage/p{i}.pkl", 'rb') #GABY CHanged
        except FileNotFoundError:
            print(f"File not found: p{i}.pkl")
            break
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
    """
    Torch Dataset for dividing a 3D volume into overlapping patches along x, y, z axes.

    Args:
        input_im (ndarray): 3D input image.
        stride (int): Step size in x and y.
        z_stride (int): Step size in z.
        quad_size (int): Patch size in x/y.
        quad_depth (int): Patch size in z.
    """
    def __init__(self, input_im, stride, z_stride, quad_size, quad_depth):
        self.im = input_im
        depth_im, width, height = input_im.shape
        self.quad_size = quad_size
        self.quad_depth = quad_depth
        xs = []
        ys = []
        zs = []
        for x in range(0,width,stride):
            if x + quad_size > width:
                difference = (x + quad_size) - width
                x = x - difference - 1
            xs.append(x)
        # print(f"x: {x}")
        for y in range(0,height,stride):
            if y + quad_size > height:
                difference = (y + quad_size) - height
                y = y - difference - 1
            # print(f"y: {y}")
            ys.append(y)
        for z in range(0,depth_im,z_stride):
            if z + quad_depth > depth_im:
                difference = (z + quad_depth) - depth_im
                z = z - difference - 1
            zs.append(z)
        xs = list(set(xs))
        ys = list(set(ys))
        zs = list(set(zs))
        xs = list(np.sort(xs))
        ys = list(np.sort(ys))
        zs = list(np.sort(zs))
        self.xs = xs
        self.ys = ys
        self.zs = zs

    def __len__(self):
        return len(self.xs) * len(self.ys) * len(self.zs)

    def __getitem__(self, idx):
        z_len = len(self.zs)
        y_len = len(self.ys)
        # map 1d indices to 3d
        z = self.zs[idx%z_len]
        y = self.ys[int(np.floor(idx/z_len))%y_len]
        x = self.xs[int(np.floor(idx / (y_len * z_len)))]
        image = self.im[z:z+self.quad_depth, x:x+self.quad_size, y:y+self.quad_size]
        image = image.transpose((1,2,0))
        coord = [z,y,x]
        return image, coord

def resolve_overlaps(csr_masks,size_threshold=20):
    """
    Resolves overlapping binary masks by identifying connected components and 
    retaining only the largest segment in each overlap group, unless overlap is small.

    Parameters:
        csr_masks (csr_matrix): Sparse mask matrix (voxels x instances).
        size_threshold (int): Threshold to ignore small overlaps.

    Returns:
        tuple:
            - List of largest retained instance indices.
            - List of removed smaller overlaps.
    """
    mask_sizes = csr_masks.sum(axis=0).A1
    overlap_matrix = csr_masks.transpose().dot(csr_masks)
    overlap_matrix.setdiag(0)
    # full_mask = np.ones(len(coo_masks.data), dtype=bool)
    largest_in_overlaps = []
    smaller_overlaps = []
    # find connected components to identify overlapping ROIs
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

# give each column an unique value
def unique_col(csr_masks):
    """
    Assigns a unique integer label to each column (instance) in a sparse mask matrix.

    Parameters:
        csr_masks (csr_matrix): Sparse binary mask matrix.

    Returns:
        csr_matrix: Mask with unique integer values per column.
    """
    column_indices = np.arange(csr_masks.shape[1])
    diagonal_matrix = diags(column_indices)
    return csr_masks.dot(diagonal_matrix)


if __name__=="__main__":
    # directory for trained model
    class Args():
        def __init__(self):
            self.dataset_name = "datasets/Rsc03_shifted_all" #GABY CHanged to match the training model
            self.exp_dir = '/cis/home/gcoste1/MaskReg/MaskRegInference/regrcnn_Rsc03_96_96_32_nms02_edited_GN_shifted4_all' #GABY CHANGED I THINK TRAINING MODEL HERE !
            self.server_env = False
    args = Args()

    # reference file for histogram matching
    # reference = skio.imread("/cis/home/gcoste1/MaskReg/RSC08/rsc03im/RSc03_roi1_t1_XTC.tif")
    # folder for temporarily storing patches
    sparse_buffer_dir ="/cis/home/gcoste1/MaskReg/MaskRegInference/InfPatchesStorage/" #GABY CHANGED for access PATCH STORAGE
    os.system(f'rm -rf {sparse_buffer_dir}*')

    # import the configuration file
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
    weight_path = onlyfiles_check[-1]
    print(f'weight_path: {weight_path}')

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
    print(f"cuda device count: {torch.cuda.device_count()}")
    net = torch.nn.DataParallel(net1)
    net.to(device)
    net.eval()

    #from csbdeep.internals import predict
    from tifffile import *

    """ Select multiple folders for analysis AND creates new subfolder for results output """
    list_folder = ["/cis/home/gcoste1/MaskReg/F12"] #GABY Folders to run here

    """ Loop through all the folders and do the analysis!!!"""
    for input_path in list_folder:
        foldername = input_path.split('/')[-2]
        sav_dir = input_path + '/' + foldername + '_output_check5'

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
        params = pd.DataFrame(columns=['filename','shape','stride','z-stride','training']) #GABY new line - make a parameter file for each folder

        # Required to initialize all
        for i in range(len(examples)):


            """ TRY INFERENCE WITH PATCH-BASED analysis from TORCHIO """
            with torch.set_grad_enabled(False):  # saves GPU RAM
                input_name = examples[i]['input']
                input_im = tiff.imread(input_name)
                # reference = skio.imread("/cis/home/zchen163/my_documents/XTC_data/Rsc03_06/RSc03_XTC_histmatch_ROI.tif")
                # input_im = match_histograms(input_im, reference) #GABY COMMENT OUT TO TEST RSC03 Trained MaskREg


                """ Analyze each block with offset in all directions """

                # Display the image
                #max_im = plot_max(input_im, ax=0)

                print('Starting inference on volume: ' + str(i) + ' of total: ' + str(len(examples)))


                overlap_percent = 0
                input_size = 96
                depth = 32
                num_truth_class = 2
                stride = 66 # 1024*1024*303 GABY STRIDE HERE
                z_stride = 15 # 1024*1024*303


                quad_size=input_size
                quad_depth=depth

                input_dataset = InferenceDataset(input_im, stride, z_stride, quad_size, quad_depth)
                print(f"input_dataset: {input_dataset.__len__()}")
                test_dataloader = torch.utils.data.DataLoader(input_dataset, batch_size=8, shuffle=False)

                # 48 groups of small patches
                skip_top=1


                thresh = 0.99
                cf.merge_3D_iou = thresh


                im_size = np.shape(input_im)
                width = im_size[1];  height = im_size[2]; depth_im = im_size[0]

                start_time = time.time()

                segmentation = np.zeros([depth_im, width, height], dtype=np.int32)
                # all_synapses = []
                all_synapses = None
                count = 0
                #%%
                # Run inference on all patches and store detected synapse locations
                for im, coord in tqdm(test_dataloader, desc="patches",leave=False):
                    # im = torch.permute(im, (0,2,3,1))  # torch 1.4.0 doesn't have permute
                    im = im[:,None,:,:,:]
                    im = im.float().to(device)
                    _, _, _, detections, detection_masks = net.module.forward(im)
                    results_dict = net.module.get_results_modified(im.shape, detections, detection_masks, return_masks=True)

                    seg_im = results_dict['masks'][np.newaxis, np.newaxis, :]
                    synapse_locs = results_dict['sparse']   # a list of tuples contains x y z coordinates
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
                #%%

                mins, secs = divmod((time.time() - start_time), 60)
                h, mins = divmod(mins, 60)
                t = "{:d}h:{:02d}m:{:02d}s".format(int(h), int(mins), int(secs))
                print("{} patch segmentation runtime: {}".format(os.path.split(__file__)[1], t))

                multi_start_time = time.time()
                
                # number of CPU processes to use
                num_processes = 64
                if count < num_processes:
                    num_processes = count
                patch_list = np.arange(num_processes).tolist()
                pool = multiprocessing.Pool(processes=num_processes)
                outputs = pool.map(partial(parallel_sparse, width=width,height=height,depth_im=depth_im,
                                           chunk_size=int(np.ceil(count/num_processes))), patch_list)
                pool.close()
                pool.join()
                all_synapses = hstack(outputs,format="coo")


                mins, secs = divmod((time.time() - multi_start_time), 60)
                h, mins = divmod(mins, 60)
                t = "{:d}h:{:02d}m:{:02d}s".format(int(h), int(mins), int(secs))
                print("{} multiprocess runtime: {}".format(os.path.split(__file__)[1], t))

                # Iteratively resolve overlapping masks to remove duplicate detections.
                csr_masks = all_synapses.tocsr()
                cleaned_masks = []
                largest_in_overlaps, smaller_overlaps = resolve_overlaps(csr_masks, size_threshold=10) # first round
                cleaned_masks.append(unique_col(csr_masks[:,largest_in_overlaps]).sum(axis=1))
                last_round = csr_masks
                while len(smaller_overlaps) > 2:
                    current_round = last_round[:,smaller_overlaps]
                    large, smaller_overlaps = resolve_overlaps(current_round, size_threshold=10)
                    cleaned_masks.append(unique_col(current_round[:,large]).sum(axis=1))
                    last_round = current_round
                merged_im = np.zeros((width, height, depth_im), dtype=np.uint32)
                for ii in range(len(cleaned_masks)):
                    current_mask = cleaned_masks[ii].A1
                    current_mask = current_mask.reshape((width, height, depth_im))
                    merged_im[current_mask>0] = current_mask[current_mask > 0]

                merged_im = np.transpose(merged_im,(2,0,1))
                labelled_im = measure.label(merged_im, connectivity=1)
                labelled_im = np.asarray(labelled_im, dtype=np.uint32)

            labelled_im = edit_labels(labelled_im)

            filename = input_name.split('/')[-1].split('.')[0:-1]
            filename = '.'.join(filename)
            # save final result
            tiff.imwrite(sav_dir + filename + '_' + str(int(i)) +'_segmentation.tif', labelled_im)

            params = params.append(dict(zip(params.columns, [filename, labelled_im.shape, stride, z_stride, args.exp_dir])), ignore_index = True) #GABY added

        params.to_pickle(sav_dir + 'MaskReg_Parameters.pkl') #GABY added
