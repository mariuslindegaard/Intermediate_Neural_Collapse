import numpy as np
from scipy.linalg import orth
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from tqdm import tqdm

from collections import OrderedDict

import pickle

import os

# dataset parameters
im_size             = 32
C                   = 10
input_ch            = 3

# optimization hyperparameters
lr                  = 0.01
lr_decay            = 0.2

epochs              = 350
epochs_lr_decay     = [epochs//3, epochs*2//3]

batch_size          = 128

momentum            = 0.9
weight_decay        = 2e-3

# analysis parameters
epoch_list          = [1,   2,   3,   4,   5,   6,   7,   8,   9,   10,   11,
                       12,  13,  14,  16,  17,  19,  20,  22,  24,  27,   29,
                       32,  35,  38,  42,  45,  50,  54,  59,  65,  71,   77,
                       85,  92,  101, 110, 121, 132, 144, 158, 172, 188,  206,
                       225, 235, 245, 268, 287, 293, 300, 313, 327, 338, 350]


class MLPBlock(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=None, use_batch_norm=False):
        super(MLPBlock, self).__init__()
        self.fc = nn.Linear(in_features=input_dim, out_features=output_dim, bias=use_bias)
        self.use_batch_norm = use_batch_norm
        if self.use_batch_norm:
            self.bn = nn.BatchNorm1d(num_features=output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        if self.use_batch_norm:
            x = self.bn(x)
        x = self.relu(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers_widths, output_size, set_etf=False, num_etf=1, use_bias=True, use_batch_norm=True):
        super(MLP, self).__init__()

        self.input_size=input_size
        self.hidden_layers_widths = hidden_layers_widths
        self.output_size=output_size
        self.etf = set_etf
        self.num_etf = max(1, min(num_etf, len(self.hidden_layers_widths)))

        layers = OrderedDict()

        layers['flatten'] = nn.Flatten()

        if len(hidden_layers_widths) == 0:
            layers['fc'] = nn.Linear(in_features=input_size, out_features=output_size, bias=use_bias)
        else:
            layers['block0'] = MLPBlock(input_dim=input_size, output_dim=hidden_layers_widths[0],
                                         use_bias=use_bias, use_batch_norm=use_batch_norm)
            for idx, (in_size, out_size) in enumerate(zip(hidden_layers_widths[:-1], hidden_layers_widths[1:])):
                layers[f'block{idx+1}'] = MLPBlock(input_dim=in_size, output_dim=out_size,
                                                    use_bias=use_bias, use_batch_norm=use_batch_norm)
            
            layers['fc'] = nn.Linear(in_features=hidden_layers_widths[-1], out_features=output_size, bias=use_bias)

            if set_etf:
                weight = torch.sqrt(torch.tensor(self.output_size/(self.output_size-1)))*(torch.eye(self.output_size)-(1/self.output_size)*torch.ones((self.output_size, self.output_size)))
                weight /= torch.sqrt((1/self.output_size*torch.norm(weight, 'fro')**2))
                layers['fc'].weight = nn.Parameter(torch.mm(weight, torch.eye(self.output_size, hidden_layers_widths[-1])))
                # layers['fc'].weight = nn.Parameter(torch.mm(weight, torch.tensor(orth(torch.randn(hidden_layers_widths[-1],self.output_size).numpy()).T)))                
                layers['fc'].weight.requires_grad_(False)
                for idx in range(len(self.hidden_layers_widths)-self.num_etf+1, len(self.hidden_layers_widths)):
                    d_out, d_in = layers[f'block{idx}'].fc.weight.shape
                    weight = torch.sqrt(torch.tensor(d_out/(d_out-1)))*(torch.eye(d_out)-(1/d_out)*torch.ones((d_out, d_out)))
                    weight /= torch.sqrt((1/d_out*torch.norm(weight, 'fro')**2))
                    layers[f'block{idx}'].fc.weight = nn.Parameter(torch.mm(weight, torch.eye(d_out,d_in)))
                    # layers[f'block{idx}'].fc.weight = nn.Parameter(torch.mm(weight, torch.tensor(orth(torch.randn(d_in,d_out).numpy()).T)))
                    layers[f'block{idx}'].fc.weight.requires_grad_(False)

            
        self.model = nn.Sequential(
            layers
        )

    def forward(self, x):
        out = self.model(x)
        return out


def measure_accuracy(model, dataloader, use_cuda=True):
    use_cuda = use_cuda and torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()
    num_samples, correct = 0, 0
    for inputs, targets in tqdm(dataloader, leave=False, desc=f'   batches: '):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        preds = model(inputs)
        correct += torch.argmax(preds, dim=-1).eq(targets).sum().item()
        num_samples += len(inputs)

    acc = correct / num_samples
    return acc
    

if __name__ == "__main__":
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    tx = transforms.Compose([transforms.ToTensor(), normalize])
    traindata = datasets.CIFAR10(root='cifar10', train=True, transform=tx)
    trainloader = DataLoader(traindata, batch_size=batch_size)
    testdata = datasets.CIFAR10(root='cifar10', train=False, transform=tx)
    testloader = DataLoader(testdata, batch_size=batch_size)

    
    layers = 10
    num_etfs = np.arange(layers-1,0,-1).tolist()
    etf = True
    l = 'mse'
    # l = 'cross_entropy'
    one_hot = True
    e = epoch_list[-1]
    accuracy = {'train_acc':{}, 'test_acc':{}}

    for n in num_etfs:
        save_dir = os.path.join('cifar_intermediate_expt_%1.3f_%d_etf'%(lr,layers-n), 'mse' if one_hot else 'cross_entropy')
        print('Loading %s : %d.pt'%(save_dir,e))
        model = MLP(input_ch * im_size**2, [1024]*layers, C, use_bias=False, set_etf=etf, num_etf=n)
        model.load_state_dict(torch.load(os.path.join(save_dir,'%d.pt'%(e)), map_location=torch.device('cpu')))
        accuracy['train_acc'][layers-n] = measure_accuracy(model, trainloader)
        accuracy['test_acc'][layers-n] = measure_accuracy(model, testloader)

    with open('cifar_etfs_accuracy.pkl', 'wb') as f:
        pickle.dump(accuracy,f)
