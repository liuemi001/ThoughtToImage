import torch
import numpy as np
import pandas as pd
import pprint

from sklearn import tree
from sklearn import ensemble
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

"""
DECISION TREES

Methods: 
- baseline for classifying eeg data by what image subjects were shown when 
  the EEG data was recorded
- Train a decision tree model on top 10 ICA components of each EEG reading

Results: 
- Baseline is not very good, around 2% accuracy for both training and validation
- EEGs are not represented very well by simply the top 10 ICA components

Questions: 
- perhaps more ICA components? or ICA is just not good
- why is test/val accuracy higher than train accuracy

"""

pp = pprint.PrettyPrinter(indent=4)

base_path = "../ThoughtToImage/DreamDiffusion-main/dreamdiffusion/datasets/"
eeg_14_70_path = base_path + "eeg_14_70_std.pth" # Likely refers to freq range
eeg_5_95_path = base_path + "eeg_5_95_std.pth"
eeg_55_95_path = base_path + "eeg_55_95_std.pth"
eeg_signals_raw_with_mean_path = base_path + "eeg_signals_raw_with_mean_std.pth"
block_splits_by_image_all_path = base_path + "block_splits_by_image_all.pth"
block_splits_by_image_path = base_path + "block_splits_by_image_single.pth"


features = np.loadtxt("ica_eeg_5_95_data.csv", delimiter=",", dtype=float)
eeg_5_95_data = torch.load(eeg_5_95_path)
labels = [data['label'] for data in eeg_5_95_data['dataset']]
labels = np.array(labels)
features = np.reshape(features, [labels.size, int(features.size/labels.size)])

# rescale 
scaler = StandardScaler()
scaler.fit(features)
features = scaler.transform(features)
print("features head: ", features[:6])


# train validation test split from block splits
# block_splits_by_image_all_data = torch.load(block_splits_by_image_all_path)
# train_indices = block_splits_by_image_all_data['splits'][0]['train']
# val_indices = block_splits_by_image_all_data['splits'][0]['val']
# test_indices = block_splits_by_image_all_data['splits'][0]['test']


# x_train = features[train_indices]
# y_train = labels[train_indices]
# x_val = features[val_indices] 
# y_val = labels[val_indices]
# x_test = features[test_indices]
# y_test = features[test_indices] 

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25)

# train on a couple different depths of decision trees
model = tree.DecisionTreeClassifier()
#model = ensemble.RandomForestClassifier()
#model = LogisticRegression(multi_class='ovr', solver='lbfgs')

print("Training...")
model.fit(x_train, y_train)

# plot results

# test on validation set
print("predicting...")
y_val_pred = model.predict(x_val)
np.savetxt("decision_tree_val_predictions.csv", y_val_pred, delimiter=",")
y_test_pred = model.predict(x_test)
np.savetxt("decision_tree_test_predictions.csv", y_test_pred, delimiter=",")
y_train_pred = model.predict(x_train)
np.savetxt("decision_tree_train_predictions.csv", y_test_pred, delimiter=",")


print("Calculating accuracy...")
print("labels: ", labels)
print("train prediction: ", y_test_pred)
print("val prediction: ", y_val_pred)
print("test prediction: ", y_test_pred)
y_val_pred = np.loadtxt("decision_tree_val_predictions.csv", delimiter=",", dtype=float).astype(int)
y_test_pred = np.loadtxt("decision_tree_test_predictions.csv", delimiter=",", dtype=float).astype(int)
val_correct = np.intersect1d(y_val, y_val_pred)
test_correct = np.intersect1d(y_test, y_test_pred)
train_correct = np.intersect1d(y_train, y_train_pred)
print("training accuracy: ", train_correct.shape[0] / y_train.shape[0])
print("accuracy val: ", val_correct.shape[0] / y_val.shape[0])
print("accuracy_test: ", test_correct.shape[0] / y_test.shape[0])
