import torch
from torch.utils.data import Dataset
import os
import numpy as np

class EEGDataset(Dataset):
    """
    A dataset class for loading and transforming EEG and corresponding ImageNet data.
    """
    def __init__(self, eeg_signals_path, imagenet_path, subject=0, time_low=20, time_high=460):
        # Load EEG signals
        loaded = torch.load(eeg_signals_path)
        self.data = [item for item in loaded['dataset'] if (item['subject'] == subject or subject == 0) and time_low <= item['eeg'].size(1) <= time_high]
        # self.data = [item for item in loaded['dataset'] if (item['subject'] == subject or subject == 0)]
        self.labels = loaded["labels"]
        self.images = loaded["images"]
        # self.imagenet = imagenet_path
        self.time_low = time_low
        self.time_high = time_high
        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        # Process EEG
        eeg = self.data[i]["eeg"].float().t()
        eeg = eeg[self.time_low:self.time_high,:]
        label = self.data[i]["label"]
        
        return eeg, label

class Splitter:
    """
    A class to split the dataset into training and testing sets.
    """
    def __init__(self, dataset, split_path, split_num=0, split_name="train"):
        self.dataset = dataset
        loaded = torch.load(split_path)
        self.split_idx = loaded["splits"][split_num][split_name]
        self.split_idx = [i for i in self.split_idx if 450 <= self.dataset.data[i]["eeg"].size(1) <= 600]
        self.size = len(self.split_idx)

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        eeg, label = self.dataset[self.split_idx[i]]
        return eeg, label

def create_EEG_dataset(eeg_signals_path, imagenet_path, splits_path, subject=0, time_low=20, time_high=460):
    """
    A function to create and split EEG dataset for training and testing.
    """
    dataset = EEGDataset(eeg_signals_path, imagenet_path, subject, time_low, time_high)
    split_train = Splitter(dataset, splits_path, split_num=0, split_name='train')
    split_test = Splitter(dataset, splits_path, split_num=0, split_name='test')
    return split_train, split_test

