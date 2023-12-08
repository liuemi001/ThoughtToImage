import torch
from torch.utils.data import Dataset
import os
import numpy as np
import random

#channels = [39, 47, 58, 66, 76, 84, 96, 100, 116, 117, 120, 121, 122, 123, 124, 125, 126, 127]
#channels = random.sample(range(128), 18)
#channels = [120, 121, 122, 123, 124, 125]  # Occipital only
#channels = [96, 100, 116, 117, 120, 121, 122, 123, 124, 125]  # Parietal and occipital
#channels = [39, 47, 58, 66, 76, 84]  # temporal channels only
#channels = [96, 100, 116, 117]  # parietal only
#channels = [39, 47, 58, 66, 76, 84, 96, 100, 116, 117]  # temporal and parietal 
#channels = [39, 58, 76, 96, 116]  # temporal and parietal left side
#channels = list(range(128))
#print("Chosen channels: ", channels)

class EEGDataset(Dataset):
    """
    A dataset class for loading and transforming EEG and corresponding ImageNet data.
    """
    def __init__(self, eeg_signals_path, channels, imagenet_path=None, subject=0, time_low=20, time_high=460):
        # Load EEG signals
        loaded = torch.load(eeg_signals_path)
        subject = 0
        
        self.data = [trim_eeg_sample(item['eeg'], channels) for item in loaded['dataset']
                              if (item['subject'] == subject or subject == 0)]
        # self.data = [item for item in loaded['dataset'] if (item['subject'] == subject or subject == 0)]
        self.labels = np.array([item['label'] for item in loaded['dataset'] 
                                if (item['subject'] == subject or subject == 0)])
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

def trim_eeg_sample(sample, channels, time_low=20, time_high=460): 
    """
    Trims a single eeg reading to a uniform time sample and removes
    irrelevant channels
    Params: 
        - sample: PyTorch Tensor 
        - channels: array specifying indices of the channels to keep
        - time_low: inclusive lower bound of timestep range to keep
        - time_high: exclusive upper bouund of timestep range to keep

    Returns: 
        - PyTorch tensor
    """
    channels = torch.tensor(channels)
    sample = torch.index_select(sample, 0, channels)
    sample = sample[:, time_low:time_high]

    return sample


class Splitter:
    """
    A class to split the dataset into training and testing sets.
    """
    def __init__(self, dataset, split_path, split_num=0, split_name="train"):
        self.dataset = dataset
        loaded = torch.load(split_path)
        self.split_idx = loaded["splits"][split_num][split_name]
        #self.split_idx = [i for i in self.split_idx if 450 <= self.dataset.data[i].size(1) <= 600]
        self.size = len(self.split_idx)

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        eeg, label = self.dataset[self.split_idx[i]]
        return eeg, label
    
    def get_split_sets(self): 
        """
        Return the train/test splits for X and y from the dataset and splits set
        stored as class variables
        """
        x_split = [self.dataset.data[i] for i in self.split_idx]
        y_split = self.dataset.labels[self.split_idx]
        return x_split, y_split

def create_EEG_dataset(eeg_signals_path, channels, splits_path, subject=0, time_low=20, time_high=460):
    """
    A function to create and split EEG dataset for training and testing.
    Returns them as numpy arrays.
    """
    dataset = EEGDataset(eeg_signals_path, channels, subject, time_low, time_high)
    split_train = Splitter(dataset, splits_path, split_num=0, split_name='train')
    x_train, y_train = split_train.get_split_sets()
    split_val = Splitter(dataset, splits_path, split_num=0, split_name='val')
    x_val, y_val = split_val.get_split_sets()
    return x_train, y_train, x_val, y_val

