#Original model presented in: C. Spampinato, S. Palazzo, I. Kavasidis, D. Giordano, N. Souly, M. Shah, Deep Learning Human Mind for Automated Visual Classification, CVPR 2017 
import sys
import os
import random
import math
import time
import torch; torch.utils.backcompat.broadcast_warning.enabled = True
from torchvision import transforms, datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.backends.cudnn as cudnn; cudnn.benchmark = True
import numpy as np

class Model(nn.Module):

    def __init__(self, input_size=128, lstm_size=128, lstm_layers=1, output_size=128):
        # Call parent
        super().__init__()
        # Define parameters
        self.input_size = input_size
        self.lstm_size = lstm_size
        self.lstm_layers = lstm_layers
        self.output_size = output_size

        # Define internal modules
        self.lstm = nn.LSTM(input_size, lstm_size, num_layers=lstm_layers, batch_first=True)
        self.output = nn.Linear(lstm_size, output_size)
        self.classifier = nn.Linear(output_size,40)
        
    def forward(self, x):
        # Prepare LSTM initiale state
        batch_size = x.shape[0]
        lstm_init = (torch.zeros(self.lstm_layers, batch_size, self.lstm_size), torch.zeros(self.lstm_layers, batch_size, self.lstm_size))
        if x.is_cuda: lstm_init = (lstm_init[0].cuda(), lstm_init[0].cuda())
        lstm_init = (Variable(lstm_init[0], volatile=x.volatile), Variable(lstm_init[1], volatile=x.volatile))

        # Forward LSTM and get final state
        x = self.lstm(x, lstm_init)[0][:,-1,:]
        
        # Forward output
        x = F.relu(self.output(x))
        x = self.classifier((x))
        return x
    
# input_size = 128  # Number of features in the input (e.g., EEG channels)
# hidden_size = 64  # Number of features in the hidden state of the LSTM
# num_layers = 2  # Number of LSTM layers
# num_classes = 40  # Number of classes for classification

# model = Model(input_size, hidden_size, num_layers, num_classes)
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# base_path = "../ThoughtToImage/DreamDiffusion-main/dreamdiffusion/datasets/"
# eeg_5_95_path = base_path + "eeg_5_95_std.pth"
# eeg_5_95_data = torch.load(eeg_5_95_path)
# all_data = eeg_5_95_data['dataset']

# batch_size = 1000
# num_exps = len(all_data)
# num_batches = int(num_exps / batch_size)

# # Training loop
# epochs = 1000
# for epoch in range(epochs):

#     for i in range(num_exps):
#         optimizer.zero_grad()
#         start = i * batch_size
#         end = min((i+1)*batch_size, num_exps)
#         input_seq = all_data[i]['eeg']
#         labels = all_data[i]['label']
#         # input_seq = np.array([all_data[start + i]['eeg'] for i in range(start, end)])
#         # labels = np.array([all_data[start + i]['label'] for i in range(start, end)])
#         output = model.forward(input_seq)  # Corrected line
#         loss = criterion(output, labels)
#         loss.backward()
#         optimizer.step()

#     if epoch % 100 == 0:
#         print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')