from sklearn import tree
import torch
import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent=4)

base_path = "../DreamDiffusion-main/dreamdiffusion/datasets/"
eeg_14_70_path = base_path + "eeg_14_70_std.pth" # Likely refers to freq range
eeg_5_95_path = base_path + "eeg_5_95_std.pth"
eeg_55_95_path = base_path + "eeg_55_95_std.pth"
eeg_signals_raw_with_mean_path = base_path + "eeg_signals_raw_with_mean_std.pth"
block_splits_by_image_all_path = base_path + "block_splits_by_image_all.pth"
block_splits_by_image_path = base_path + "block_splits_by_image_single.pth"

# load all feature and label data
eeg_14_70_data = torch.load(eeg_14_70_path)

features = [data['eeg'] for data in eeg_14_70_data['dataset']]
print(features)
#features = np.array(features)  # TODO: cannot coerce into a numpy array currently...

labels = eeg_14_70_data['labels']

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


# compress tensors into a reasonable number of features

# train on a couple different depths of decision trees
clf = tree.DecisionTreeClassifier()

# plot results

# test on validation set

# pick best tree

# evaluate on test set

