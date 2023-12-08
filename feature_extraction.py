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

def extract_features(eeg_sample):
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
    mean = torch.mean(eeg_sample)
    variance = torch.var(eeg_sample)
    features += [mean.item(), variance.item()]
    
    # frequency-domain features (Welch's method)
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
    print("Extracting pyeeg features...")
    num_channels = eeg_sample.size(0)
    for i in range(num_channels):
        channel = eeg_sample[i].numpy()
        hjorth_mobility, hjorth_complexity = pyeeg.hjorth(channel)
        hurst = pyeeg.hurst(channel)
        tau, de = 2, 20
        svd_entropy = pyeeg.svd_entropy(channel, tau, de)
        features += [hjorth_mobility, hjorth_complexity, hurst, svd_entropy]

    #features = [mean.item(), variance.item()] + list(band_powers.values())
    return features

def construct_dataset(data):
    X = [extract_features(eeg_data['eeg']) for eeg_data in data['dataset']]
    y = [entry['label'] for entry in data['dataset']]
    return X, y

def define_model():
    # Using a simple RandomForest model as an example. You can modify this.
    model = RandomForestClassifier()
    return model

def train_and_evaluate(X_train, y_train, X_test, y_test, model):
    model.fit(X_train, y_train)
    train_predictions = model.predict(X_train)
    val_predictions = model.predict(X_test)
    train_accuracy = accuracy_score(y_train, train_predictions)
    val_accuracy = accuracy_score(y_test, val_predictions)
    return train_accuracy, val_accuracy

def main():
    # Load and split data using channel selection, pregiven splits
    print("Creating splits and extracting features...")
    X_train, y_train, X_val, y_val = dataset.create_EEG_dataset(eeg_55_95_path, "", block_splits_by_image_all_path)
    X_train = [extract_features(x) for x in X_train]
    X_val  = [extract_features(x) for x in X_val]

    # Define model
    model = define_model()
    
    # Train and evaluate
    print("Training...")
    train_accuracy, val_accuracy = train_and_evaluate(X_train, y_train, X_val, y_val, model)
    print(f"Train Accuracy: {train_accuracy:.2f}")
    print(f"Val Accuracy: {val_accuracy:.2f}")

if __name__ == "__main__":
    main()