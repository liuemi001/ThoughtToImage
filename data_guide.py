import torch

eeg_14_70_path = "C:/Users/aryaa/Downloads/data/eeg_14_70_std.pth" # Likely refers to freq range
eeg_5_95_path = "C:/Users/aryaa/Downloads/data/eeg_5_95_std.pth"
eeg_55_95_path = "C:/Users/aryaa/Downloads/data/eeg_55_95_std.pth"
eeg_signals_raw_with_mean_path = "C:/Users/aryaa/Downloads/data/eeg_signals_raw_with_mean_std.pth"
block_splits_by_image_all_path = "C:/Users/aryaa/Downloads/data/block_splits_by_image_all.pth"
block_splits_by_image_path = "C:/Users/aryaa/Downloads/data/block_splits_by_image_single.pth"

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
# 
# 
# eeg_signals_raw_with_mean_data
# type: dict 
# dict keys: ['dataset', 'labels', 'images', 'means', 'stddevs'] <-- same inner structure as before too
# 
# block_splits_by_image_all
# purpose: splits EEG data into training, validation, and test sets 
# type: dict
# dict keys: split 
# [{}, {}, {}, {}, {}, {}]
# Each {} contains training, validation, testing



import pprint
pp = pprint.PrettyPrinter(indent=4)

# ** eeg_14_70 (same can be done for eeg_5_95, eeg_55_95) ** 
# eeg_14_70_data = torch.load(eeg_14_70_path)
# pp.pprint(eeg_14_70_data['dataset'][:5])  # First 5 entries of the 'dataset'
# print("Labels sample:", eeg_14_70_data['labels'][:5])  # First 5 labels
# print("Images sample:", daeeg_14_70_datata['images'][:5])  # First 5 images

# ** eeg_signals_raw_with_mean ** 
# eeg_signals_raw_with_mean_data = torch.load(eeg_signals_raw_with_mean)
# pp.pprint(eeg_signals_raw_with_mean_data['dataset'][:5])
# print(eeg_signals_raw_with_mean_data['means'][0])

# ** block_splits_by_image_all ** 
block_splits_by_image_all_data = torch.load(block_splits_by_image_all_path)
print(block_splits_by_image_all_data['splits'][0]['train'])
