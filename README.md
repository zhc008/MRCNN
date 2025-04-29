The RegRCNN is a forked version of https://github.com/MIC-DKFZ/RegRCNN
## Installation and compatibility
For installation of all the packages used, Python 3.7.5 is required.  Example installation of Python 3.7.5:
```
wget https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz
tar zxfv Python-3.7.5.tgz
rm Python-3.7.5.tgz
find ./Python-3.7.5/Python -type d | xargs chmod 0755
cd Python-3.7.5
./configure --prefix=$PWD/Python-3.7.5/Python
make
make install
export PATH=$CWD:$PATH
```
Then create a virtual environment from inside the folder of Python-3.7.5:
```
cd Python-3.7.5/
python -m venv /your_dir_to_RegRCNN/RegRCNN/pyenv
cd /your_dir_to_RegRCNN/RegRCNN/
source pyenv/bin/activate
python setup.py install
```
## Test case
Run inference on a small 96x96x32 image in folder RSC02_roi3_GC
```
python evaluate_inference_analysis_masks_small.py
```
inference result will be under the RSC02_roi3_GC folder

## Train
To train a model, create your own dataset and config file.  
Example file for creating dataset is ./RegRCNN/datasets/Rsc03_shifted_all/preprocessing_shifted4.py  
Example config file is ./RegRCNN/datasets/config.py  
run the exec.py file for training
```
python exec.py --mode train --dataset_name your_dataset --exp_dir path/to/experiment/directory
```
For more detailed description, look at the README file in the RegRCNN folder.

## Run Inference file
Open the script inference_sparse_parallel_GC.py  
Change the dir of the training model folder in self.dataset_name, and the directory of the trained model in self.exp_dir at line 206-207  
```
self.dataset_name = ""
self.exp_dir = ''
```
Set the directory for storing intermediate output at line 214:
```
sparse_buffer_dir=""
```
Change the dir path in the parallel_sparse() function accordingly at line 103:
```
file = open(f"your_sparse_buffer_dir/p{i}.pkl/", 'rb')
```
Set the list of folders storing inference images at line 286
```
list_folder = [""]
```
Set the save_dir name at line 290-291
```
foldername = input_path.split('/')[-2]
sav_dir = input_path + '/' + foldername + '_output'
```
Set the stride for cutting the large image into small patches at line 328-333:
```
overlap_percent = 
input_size = 
depth = 
num_truth_class = 
stride = 
z_stride = 
```
Set the num_processes based on number of cpus for multiprocessing at line 389
```
num_processes = 
```
Run the parallel inference file
```
python inference_sparse_parallel_GC.py
```