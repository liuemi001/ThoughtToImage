import dataset
import pyeeg
import torch
import numpy as np

base_path = "../ThoughtToImage/DreamDiffusion-main/dreamdiffusion/datasets/"
eeg_14_70_path = base_path + "eeg_14_70_std.pth" # Likely refers to freq range
eeg_5_95_path = base_path + "eeg_5_95_std.pth"
eeg_55_95_path = base_path + "eeg_55_95_std.pth"
eeg_signals_raw_with_mean_path = base_path + "eeg_signals_raw_with_mean_std.pth"
block_splits_by_image_all_path = base_path + "block_splits_by_image_all.pth"
block_splits_by_image_path = base_path + "block_splits_by_image_single.pth"


class ChannelSelection(object): 

    def __init__(self, eeg_signals_path, splits_path):
        # only use training data for running channel selection algorithms
        self.eeg_data, self.labels, _, _ = dataset.create_EEG_dataset(eeg_signals_path, "", splits_path)

        self.num_channels = self.eeg_data[0].size(0)

        self.temporal_channels_18 = [39, 47, 58, 66, 76, 84]
        self.parietal_channels_18 = [96, 100, 116, 117]
        self.occipital_channels_18 = [120, 121, 122, 123, 124, 125]

    
    def select_by_svd_entropy(self, N=18): 
        """
        Select top N channels by greatest svd entropy. 

        Params: 
            - N: (optional) number of channels to select. default 18. 
        Returns: 
            - top N channels with the greatest svd entropy across all training examples in order
        """
        channel_entropies = np.array([])
        tau, de = 2, 20

        for channel in range(self.num_channels): 
            vals = np.array([pyeeg.svd_entropy(sample[channel], tau, de) for sample in self.eeg_data])
            channel_entropies = np.append(channel_entropies, np.mean(vals))

        channels_most_to_least = np.argsort(channel_entropies)[::-1]

        return channels_most_to_least[:N]


    def select_by_spectral_entropy(self, N=18): 
        """
        Select top N channels by greatest spectral entropy. 

        Params: 
            - N: (optional) number of channels to select. default 18. 
        Returns: 
            - top N channels with the greatest spectral entropy across all training examples in order
        """
        channel_entropies = np.array([])
        band = [1, 4, 8, 12, 30, 45]
        Fs = 250

        for channel in range(self.num_channels): 
            pow_ratios = np.array([pyeeg.bin_power(sample[channel], band, Fs)[1] for sample in self.eeg_data])
            power_ratio = np.mean(pow_ratios, axis=0)
            vals = np.array([pyeeg.spectral_entropy(sample[channel], band, Fs, power_ratio) for sample in self.eeg_data])
            channel_entropies = np.append(channel_entropies, np.mean(vals))

        channels_most_to_least = np.argsort(channel_entropies)[::-1]

        return channels_most_to_least[:N]
        
    
    def select_by_tot_variance(self, N=18): 
        """
        Select top N channels by greatest variance. 

        Params: 
            - N: (optional) number of channels to select. default 18. 
        Returns: 
            - top N channels with the greatest variance across all training examples in order
        """
        channel_vars = np.array([])

        for channel in range(self.num_channels): 
            vars = np.array([torch.var(sample[channel]) for sample in self.eeg_data])
            channel_vars = np.append(channel_vars, np.mean(vars))

        channels_most_to_least = np.argsort(channel_vars)[::-1]

        return channels_most_to_least[:N]
        

    def select_18_by_physio(self): 
        return self.temporal_channels_18 + self.parietal_channels_18 + self.occipital_channels_18
        


def main(): 
    selector = ChannelSelection(eeg_55_95_path, block_splits_by_image_all_path)
    print("Calculating SVD Entropies...")
    svd_channels = selector.select_by_svd_entropy()
    print("Calculating spectral entropies...")
    spec_channels = selector.select_by_spectral_entropy()
    print("Calculating variances...")
    var_channels = selector.select_by_tot_variance()

    print("Channels selected by SVD Entropy: ", svd_channels)
    print("Channels selected by spectral entropy: ", spec_channels)
    print("Channels selected by variance: ", var_channels)

if __name__ == '__main__': 
    main()