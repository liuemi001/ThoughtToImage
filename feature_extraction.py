import torch
import numpy as np
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pprint import pprint
import dataset

base_path = "../ThoughtToImage/DreamDiffusion-main/dreamdiffusion/datasets/"
eeg_14_70_path = base_path + "eeg_14_70_std.pth" # Likely refers to freq range
eeg_5_95_path = base_path + "eeg_5_95_std.pth"
eeg_55_95_path = base_path + "eeg_55_95_std.pth"
eeg_signals_raw_with_mean_path = base_path + "eeg_signals_raw_with_mean_std.pth"
block_splits_by_image_all_path = base_path + "block_splits_by_image_all.pth"
block_splits_by_image_path = base_path + "block_splits_by_image_single.pth"

def extract_features(eeg_sample):
    # time-domain features
    # mean = eeg_sample.mean()
    # variance = eeg_sample.var()
    mean = torch.mean(eeg_sample)
    variance = torch.var(eeg_sample)
    
    # frequency-domain features (Welch's method)
    freqs, psd = welch(eeg_sample.numpy(), fs=250) # sampling frequency of 250Hz, can up to 1000Hz # can remove .numpy()
    avg_psd = np.mean(psd, axis=0)

    band_powers = {}
    bands = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 12), 'beta': (12, 30), 'gamma': (30, 45)}
    for band, (low_freq, high_freq) in bands.items():
        idx_band = np.logical_and(freqs >= low_freq, freqs <= high_freq)
        band_power = avg_psd[idx_band].sum()
        band_powers[band] = band_power

    features = [mean.item(), variance.item()] + list(band_powers.values())
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
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

def main():
    # Load data
    data = torch.load(eeg_55_95_path)
    print(data.keys())
    print(data['dataset'][0])
    print(data['labels'][0])
    print(data['images'][0])
    
    # Construct dataset
    X, y = construct_dataset(data)
    
    # Split data
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Split data using channel selection, pregiven splits
    X_train, y_train, X_val, y_val = dataset.create_EEG_dataset(eeg_55_95_path, "", block_splits_by_image_all_path)
    X_train = [extract_features(x) for x in X_train]
    print("len x_Train: ", len(X_train))
    print("len y train: ", len(y_train))
    X_val  = [extract_features(x) for x in X_val]
    print("len x val: ", len(X_val))
    print("len y val: ", len(y_val))

    # Define model
    model = define_model()
    
    # Train and evaluate
    accuracy = train_and_evaluate(X_train, y_train, X_val, y_val, model)
    print(f"Val Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()