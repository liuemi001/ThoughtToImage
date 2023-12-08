import torch
import numpy as np
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from pprint import pprint
import dataset
import pyeeg 

base_path = "../ThoughtToImage/DreamDiffusion-main/dreamdiffusion/datasets/"
eeg_14_70_path = base_path + "eeg_14_70_std.pth" # Likely refers to freq range
eeg_5_95_path = base_path + "eeg_5_95_std.pth"
eeg_55_95_path = base_path + "eeg_55_95_std.pth"
eeg_signals_raw_with_mean_path = base_path + "eeg_signals_raw_with_mean_std.pth"
block_splits_by_image_all_path = base_path + "block_splits_by_image_all.pth"
block_splits_by_image_path = base_path + "block_splits_by_image_single.pth"

class VisualClassifer(object): 

    def __init__(self, eeg_path, splits_path, model=None, time_feats=True, pyeeg_feats=True):
        if model != None: 
            self.model = model
        else: 
            self.model = RandomForestClassifier()  # model to use. default to RandomForest

        # paths to data
        self.eeg_path = eeg_path
        self.splits_path = splits_path

        # datasets. will be filled in construct_dataset function
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None

        # booleans indicating which types of features you want to use
        self.time_feats = time_feats
        self.pyeeg_feats = pyeeg_feats
    

    def extract_features(self, eeg_sample):
        """
        Extract features like mean, variance, power in different frequency bands, [stuff from pyeeg] from
        a single EEG reading
        Params: 
            - eeg_sample: PyTorch tensor that represents one eeg sample with varying number of channels and
                        captured timesteps
        Returns: 
            - list of features corresponding to the eeg sample
        """
        features = []

        # time-domain features
        if self.time_feats: 
            mean = torch.mean(eeg_sample)
            variance = torch.var(eeg_sample)
            features += [mean.item(), variance.item()]
        
        # frequency-domain features (Welch's method)
        # if using pyeeg features, we are already calculating channel specific power and don't need 
        # to add redundant average power features
        if not self.pyeeg_feats: 
            freqs, psd = welch(eeg_sample.numpy(), fs=250) # sampling frequency of 250Hz, can up to 1000Hz # can remove .numpy()
            avg_psd = np.mean(psd, axis=0)

            band_powers = {}
            bands = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 12), 'beta': (12, 30), 'gamma': (30, 45)}
            for band, (low_freq, high_freq) in bands.items():
                idx_band = np.logical_and(freqs >= low_freq, freqs <= high_freq)
                band_power = avg_psd[idx_band].sum()
                band_powers[band] = band_power
            features += list(band_powers.values())

        # pyeeg features
        if self.pyeeg_feats: 
            num_channels = eeg_sample.size(0)
            for i in range(num_channels):
                channel = eeg_sample[i].numpy()
                hjorth_mobility, hjorth_complexity = pyeeg.hjorth(channel)
                hurst = pyeeg.hurst(channel)
                tau, de = 2, 20
                svd_entropy = pyeeg.svd_entropy(channel, tau, de)
                pfd = pyeeg.pfd(channel)
                power, power_ratio = pyeeg.bin_power(channel, [1, 4, 8, 12, 30, 45], 250)  # freq bins, samp freq of 250 Hz
                features += [hjorth_mobility, hjorth_complexity, hurst, svd_entropy, pfd] + list(power) 

        return features

    def construct_dataset(self, channels):
        self.X_train, self.y_train, self.X_val, self.y_val = dataset.create_EEG_dataset(self.eeg_path, channels, self.splits_path)
        self.X_train = [self.extract_features(x) for x in self.X_train]
        self.X_val  = [self.extract_features(x) for x in self.X_val]


    def train_and_evaluate(self):
        self.model.fit(self.X_train, self.y_train)
        train_predictions = self.model.predict(self.X_train)
        val_predictions = self.model.predict(self.X_val)
        train_accuracy = accuracy_score(self.y_train, train_predictions)
        val_accuracy = accuracy_score(self.y_val, val_predictions)
        return train_accuracy, val_accuracy
    

def main():
    # Load channels selected separately using ChannelSelector
    svd_channels = [106, 85, 55, 52, 102, 32, 35, 98, 36, 71, 73, 72, 101, 105, 69, 67, 70, 45]
    spectral_channels = [102, 100, 65, 19, 64, 24, 18, 25, 13, 20, 52, 22, 29, 37, 27, 26, 80, 17]
    var_channels = [127, 50, 97, 102, 32, 35, 123, 101, 107, 36, 98, 126, 112, 45, 73, 106, 6, 119]

    # Load and split data using channel selection, pregiven splits
    print("Initializing model...")
    clf = VisualClassifer(eeg_55_95_path, block_splits_by_image_all_path)

    # Train and evaluate on SVD Channels
    print("Extracting SVD Channel features...")
    clf.construct_dataset(svd_channels)
    print("Training on SVD Channels...")
    train_accuracy, val_accuracy = clf.train_and_evaluate()
    print(f"Train Accuracy: {train_accuracy:.2f}")
    print(f"Val Accuracy: {val_accuracy:.2f}")

    # Train and evaluate on spec Channels
    print("Extracting SVD Channel features...")
    clf.construct_dataset(spectral_channels)
    print("Training on Spectral Entropy Channels...")
    train_accuracy, val_accuracy = clf.train_and_evaluate()
    print(f"Train Accuracy: {train_accuracy:.2f}")
    print(f"Val Accuracy: {val_accuracy:.2f}")

    # Train and evaluate on variance Channels
    print("Extracting SVD Channel features...")
    clf.construct_dataset(var_channels)
    print("Training on Variance Channels...")
    train_accuracy, val_accuracy = clf.train_and_evaluate()
    print(f"Train Accuracy: {train_accuracy:.2f}")
    print(f"Val Accuracy: {val_accuracy:.2f}")

if __name__ == "__main__":
    main()