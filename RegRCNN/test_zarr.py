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
from scipy.sparse import coo_matrix, hstack
from scipy import sparse
import zarr

profiler = cProfile.Profile()
profiler.enable() 

width, height, depth = 1024, 1024, 303
merged_im = zarr.open('example.zarr', mode='w', shape=(width, height, depth), chunks=(48,48,3), dtype='uint32')
all_synapses = sparse.load_npz("all_synapses.npz")
size1d, num_synapses = all_synapses.shape[0], all_synapses.shape[1]
# synapses_csr = all_synapses.tocsr()
for col in tqdm(range(num_synapses),desc='synpases', leave=False):
    col_mask = all_synapses.col == col
    non_zero_rows = all_synapses.row[col_mask]
    for i in non_zero_rows:
        x = i // (height * depth)
        y = (i % (height * depth)) // depth
        z = (i % (height * depth)) % depth
        merged_im[x,y,z] = col + 1
numpy_merged = np.array(merged_im)
skio.imsave("merged_dense_synapses_zarr.tif", numpy_merged)


profiler.disable() # end profiler

# Create an output file
output_file = "profiling_merge_sparse_zarr.txt"
with open(output_file, 'w') as f:
    stats = io.StringIO()
    ps = pstats.Stats(profiler, stream=stats)
    ps.sort_stats('time')  # Sort statistics by time taken
    ps.print_stats()
    f.write(stats.getvalue())
