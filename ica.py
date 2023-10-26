import torch
import numpy as np
import pprint
from sklearn.decomposition import FastICA

pp = pprint.PrettyPrinter(indent=4)

"""
Script to run ICA on each training eeg samples and compress to small number of independent components
for future training
"""

base_path = "../ThoughtToImage/DreamDiffusion-main/dreamdiffusion/datasets/"
eeg_14_70_path = base_path + "eeg_14_70_std.pth" # Likely refers to freq range
eeg_5_95_path = base_path + "eeg_5_95_std.pth"
eeg_55_95_path = base_path + "eeg_55_95_std.pth"
eeg_signals_raw_with_mean_path = base_path + "eeg_signals_raw_with_mean_std.pth"
block_splits_by_image_all_path = base_path + "block_splits_by_image_all.pth"
block_splits_by_image_path = base_path + "block_splits_by_image_single.pth"

# load all feature and label data
eeg_5_95_data = torch.load(eeg_5_95_path)

features = [data['eeg'] for data in eeg_5_95_data['dataset']]
features = [data.cpu().numpy() for data in features]  # convert each eeg sample to a numpy array
print("total number of samples: ", len(features))

### ICA ###
# ICA for each of the eeg tensors to compress into a reasonable number of features

reduced_feats = np.array([])
i = 0
for data in features: 
    ica = FastICA(n_components=10)  
    transformed = ica.fit_transform(data)
    reduced_feats = np.append(reduced_feats, transformed, axis=0)
    i += 1
    print("iteration: ", i)

np.savetxt("ica_eeg_5_95_data.csv", reduced_feats, delimiter=",")

features = reduced_feats
