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
