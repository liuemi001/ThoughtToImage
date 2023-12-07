# ThoughtToImage
CS 229 Final Project - Thought to Image: Using Brain EEG Signals to Generate Images 

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
