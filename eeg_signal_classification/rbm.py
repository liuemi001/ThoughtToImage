import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import make_grid , save_image
import matplotlib.pyplot as plt

"""
RESTRICTED BOLTZMANN MACHINE (RBM)

Purpose: 
- learn a representation of EEG data using RBM 

Background: 
- Deef Belief Networks (DBNs) and ConvNets have been used to learn representations of fMRI
  and EEG in Plis et al., 2014; Mirowski et al., 2009. 
- RBMs are one of the base unit of DBNs. DBNs can be thought of as a stack of RBMs, plus some other stuff. 
- Adding RBMs to deep belief network improves representations and accuracy (Plis et al., 2014)

Downstream: 
- Use RBM representation to train a decision tree to classify EEG data based on what image the
  subject was looking at when the readings were recorded

"""

class RBM(nn.Module):
   def __init__(self,
               n_vis=784,
               n_hin=500,
               k=5):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hin,n_vis)*1e-2)
        self.v_bias = nn.Parameter(torch.zeros(n_vis))
        self.h_bias = nn.Parameter(torch.zeros(n_hin))
        self.k = k
    
   def sample_from_p(self,p):
       return F.relu(torch.sign(p - Variable(torch.rand(p.size()))))
    
   def v_to_h(self,v):
        p_h = F.sigmoid(F.linear(v,self.W,self.h_bias))
        sample_h = self.sample_from_p(p_h)
        return p_h,sample_h
    
   def h_to_v(self,h):
        p_v = F.sigmoid(F.linear(h,self.W.t(),self.v_bias))
        sample_v = self.sample_from_p(p_v)
        return p_v,sample_v
        
   def forward(self,v):
        pre_h1,h1 = self.v_to_h(v)
        
        h_ = h1
        for _ in range(self.k):
            pre_v_,v_ = self.h_to_v(h_)
            pre_h_,h_ = self.v_to_h(v_)
        
        return v,v_
    
   def free_energy(self,v):
        vbias_term = v.mv(self.v_bias)
        wx_b = F.linear(v,self.W,self.h_bias)
        hidden_term = wx_b.exp().add(1).log().sum(1)
        return (-hidden_term - vbias_term).mean()
   

# def main(): 
rbm = RBM(n_vis=500, n_hin=250, k=1)
train_op = optim.SGD(rbm.parameters(),0.1)

base_path = "../ThoughtToImage/DreamDiffusion-main/dreamdiffusion/datasets/"
eeg_5_95_path = base_path + "eeg_5_95_std.pth"
eeg_5_95_data = torch.load(eeg_5_95_path)
all_data = eeg_5_95_data['dataset']
batch_size = 1000
num_exps = len(all_data)
num_batches = int(num_exps / batch_size)
print("number training examples: ", num_exps)


for epoch in range(10):
    loss_ = []
    for i in range(num_exps):  # TODO: change to our data
        # data.view(-1, 784) is making it be size batch size, flattened vector for each image
        #data = Variable(data.view(-1,784))  # Variable is like a tensor
        #sample_data = data.bernoulli()
        start = i * batch_size
        end = min((i+1)*batch_size, num_exps)
        # data = [all_data[start + i]['eeg'] for i in range(start, end)]  # this should already be in tensor form
        # data = torch.Tensor(data)
        # data = Variable(data.view(-1, 64000))
        data = all_data[i]['eeg']
        #target = all_data[start:end]['label']
        
        v,v1 = rbm(data)
        loss = rbm.free_energy(v) - rbm.free_energy(v1)
        loss_.append(loss.data)
        train_op.zero_grad()
        loss.backward()
        train_op.step()

    print("Training loss for {} epoch: {}".format(epoch, np.mean(loss_)))

if __name__ == 'main':
    main()