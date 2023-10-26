import torch
import numpy as np
import pandas as pd
import pprint

from sklearn import tree
from sklearn.decomposition import FastICA
from sklearn.metrics import accuracy_score


pp = pprint.PrettyPrinter(indent=4)

base_path = "../ThoughtToImage/DreamDiffusion-main/dreamdiffusion/datasets/"
eeg_14_70_path = base_path + "eeg_14_70_std.pth" # Likely refers to freq range
eeg_5_95_path = base_path + "eeg_5_95_std.pth"
eeg_55_95_path = base_path + "eeg_55_95_std.pth"
eeg_signals_raw_with_mean_path = base_path + "eeg_signals_raw_with_mean_std.pth"
block_splits_by_image_all_path = base_path + "block_splits_by_image_all.pth"
block_splits_by_image_path = base_path + "block_splits_by_image_single.pth"


features = np.loadtxt("ica_eeg_14_70_data.csv", delimiter=",", dtype=float)
eeg_5_95_data = torch.load(eeg_5_95_path)
labels = np.array([eeg_5_95_data['labels']])
features.reshape(labels.size, features.size/labels.size)



# train validation test split
block_splits_by_image_all_data = torch.load(block_splits_by_image_all_path)
train_indices = block_splits_by_image_all_data['splits'][0]['train']
val_indices = block_splits_by_image_all_data['splits'][0]['val']
test_indices = block_splits_by_image_all_data['splits'][0]['test']

x_train = features[train_indices]
y_train = labels[train_indices]
x_val = features[val_indices]
y_val = labels[val_indices]
x_test = features[test_indices]
y_test = features[test_indices]

# train on a couple different depths of decision trees
model = tree.DecisionTreeClassifier()
model.fit(x_train, y_train)

# plot results

# test on validation set
y_val_pred = model.predict(x_val)
y_test_pred = model.predict(x_test)

accuracy_val = accuracy_score(y_val.cpu().numpy(), y_val_pred)
accuracy_test = accuracy_score(y_test.cpu().numpy(), y_test_pred)
print("accuracy val: ", accuracy_val)
print("accuracy_test: ", accuracy_test)

# pick best tree

# evaluate on test set

