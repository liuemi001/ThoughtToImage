from dataset import EEGDataset
import pyeeg

base_path = "../ThoughtToImage/DreamDiffusion-main/dreamdiffusion/datasets/"
eeg_14_70_path = base_path + "eeg_14_70_std.pth" # Likely refers to freq range
eeg_5_95_path = base_path + "eeg_5_95_std.pth"
eeg_55_95_path = base_path + "eeg_55_95_std.pth"
eeg_signals_raw_with_mean_path = base_path + "eeg_signals_raw_with_mean_std.pth"
block_splits_by_image_all_path = base_path + "block_splits_by_image_all.pth"
block_splits_by_image_path = base_path + "block_splits_by_image_single.pth"


class ChannelSelection(object): 

    def __init__(self, eeg_signals_path):
        dataset = EEGDataset(eeg_signals_path)
        self.eeg_data = dataset.data
        self.labels = dataset.labels

        self.temporal_channels_18 = [39, 47, 58, 66, 76, 84]
        self.parietal_channels_18 = [96, 100, 116, 117]
        self.occipital_channels_18 = [120, 121, 122, 123, 124, 125]

    
    def select_by_entropy(self): 
        pass

    def select_18_by_physio(self): 
        return self.temporal_channels_18 + self.parietal_channels_18 + self.occipital_channels_18
        


def main(): 
    selector = ChannelSelection(eeg_55_95_path)

if __name__ == 'main': 
    main()