import torch
import numpy as np
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pprint import pprint

# ** How Everything Is Formatted **
# eeg_14_70_path
    # type: dict
    # dict keys: dataset, labels, images
    #
    # dataset[0]
    # {   'eeg': tensor([[-1.6570e-02,  2.7305e-02,  8.0402e-02,  ..., -7.7733e-05,
    #          -1.9835e-02, -1.5594e-02],
    #         [-1.0071e-02,  1.1867e-01,  2.4788e-01,  ...,  1.8917e-02,
    #          -8.7614e-03, -1.9201e-02],
    #         [ 2.5684e-02, -1.7518e-01, -3.8919e-01,  ..., -1.4881e-01,
    #          -5.8758e-02,  1.8632e-02],
    #         ...,
    #         [ 1.6717e-02,  4.4144e-02,  7.0098e-02,  ...,  1.6365e-01,
    #           8.8041e-02, -7.7776e-03],
    #         [-1.6673e-03, -4.8546e-03, -7.2816e-03,  ...,  5.2610e-03,
    #           1.6651e-03, -2.9540e-03],
    #         [ 6.4596e-03,  3.2524e-02,  6.4534e-02,  ...,  8.3441e-02,
    #           3.8469e-02, -1.5483e-02]]), 
    #     'image': 0, 
    #     'label': 10,
    #     'subject': 4}
    #  eeg seems to represent EEG readings, image indicates index of associated image, label indicates category of image
    #
    # labels[:5]
    # ['n02389026', 'n03888257', 'n03584829', 'n02607072', 'n03297495']
    # 
    # images[:5]
    # ['n02951358_31190', 'n03452741_16744', 'n04069434_10318', 'n02951358_34807', 'n03452741_5499']

# eeg_signals_raw_with_mean_data
    # type: dict 
    # dict keys: ['dataset', 'labels', 'images', 'means', 'stddevs'] <-- same inner structure as before too

# block_splits_by_image_all
    # purpose: splits EEG data into training, validation, and test sets 
    # type: dict
    # dict keys: split 
    # [{}, {}, {}, {}, {}, {}]
    # Each {} contains training, validation, testing

base_path = "../ThoughtToImage/DreamDiffusion-main/datasets/"
eeg_14_70_path = base_path + "eeg_14_70_std.pth" # Likely refers to freq range
eeg_5_95_path = base_path + "eeg_5_95_std.pth"
eeg_55_95_path = base_path + "eeg_55_95_std.pth"
eeg_signals_raw_with_mean_path = base_path + "eeg_signals_raw_with_mean_std.pth"
block_splits_by_image_all_path = base_path + "block_splits_by_image_all.pth"
block_splits_by_image_path = base_path + "block_splits_by_image_single.pth"

def extract_features(eeg_sample):
    # time-domain features
    mean = eeg_sample.mean()
    variance = eeg_sample.var()
    # mean = torch.mean(eeg_sample)
    # variance = torch.var(eeg_sample)
    
    # frequency-domain features (Welch's method)
    freqs, psd = welch(eeg_sample, fs=250)
    # freqs, psd = welch(eeg_sample.numpy(), fs=250)  # sampling frequency of 250Hz, can up to 1000Hz
    
    # print(freqs.shape)
    # print(psd.shape)
    
    #alpha_power = psd[(freqs >= 8) & (freqs <= 12)].sum()  # Example for alpha band

    # You can add more features if needed.
    
    return [mean.item(), variance.item()] #, alpha_power]

def construct_dataset(data):
    X = [extract_features(eeg_data['eeg'].numpy()) for eeg_data in data['dataset']]
    y = [entry['label'] for entry in data['dataset']]
    return X, y
    # data = torch.load(data_path)
    # X = [extract_features(eeg_data['eeg'].numpy()) for eeg_data in data['dataset']]
    # y = [entry['label'] for entry in data['dataset']]
    # return X, y

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
    # print(construct_dataset(data))
    # pprint(X)
    # pprint(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define model
    model = define_model()
    
    # Train and evaluate
    accuracy = train_and_evaluate(X_train, y_train, X_test, y_test, model)
    print(f"Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()