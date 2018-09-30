This folder contains the extracted HF0 and CQT features for each track in the dataset. Note that the files are in H5PY format and you can reach the data with 'HF0' or 
'cqt' tag. 

Example:

import h5py
import numpy as np

feats = h5py.File('AClassicEducation_NightOwl.h5','r') 
HF0 = np.array(feats['HF0'])
