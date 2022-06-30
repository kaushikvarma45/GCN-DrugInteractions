import torch
import pandas as pd
import numpy as np
import torch_geometric.transforms as T
import torch_geometric
from torch_geometric.data import Dataset
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_adj
import numpy as np 
import os
import deepchem as dc
import torch.nn as nn
from torch.nn import Linear,BatchNorm1d
from torch_geometric.nn import GCNConv,GATConv
from torch_geometric.nn import BatchNorm
from tqdm import tqdm
import pandas as pd 

import math
import warnings
warnings.filterwarnings("ignore")

class NumbersDataset(Dataset):
    def __init__(self,filename,dictval):
        datat = pd.read_csv(filename)
        # datat = datat.head(partition) 
        self.samples = list(range(2*len(datat)))
        self.label = list(range(2*len(datat)))
        self.sz = 0
        idx = 0
        for index in range(len(datat)) :
          str1 = datat['Drug1_ID'][index]
          str2 = datat['Drug2_ID'][index]

          if (str1 in dictval) and (str2 in dictval) :
            loop = 0
            if datat['label'][index]==0 :
              loop = 0
            elif datat['label'][index]==1 :
              loop = 0
            else :
              loop = 0

            mat1 = dictval[str1][0]
            mat2 = dictval[str2][0]
            mean1 = torch.mean(mat1, axis = 0)
            std1 = torch.std(mat1, axis = 0)
            mean2 = torch.mean(mat2, axis = 0)
            std2 = torch.std(mat2, axis = 0)
            vect = torch.cat([mean1, mean2, std1, std2], dim=0)
            self.samples[idx] = vect.detach().cpu()
            self.label[idx] = datat['label'][index]
            self.label[idx] = torch.tensor(self.label[idx], dtype=torch.int64)
            idx+=1
            for genvar in range(0,loop) : 
              eps = torch.randn_like(dictval[str1][1])
              std = torch.exp(dictval[str1][1])
              mat1 = eps.mul(std).add_(dictval[str1][0])
              eps = torch.randn_like(dictval[str2][1])
              std = torch.exp(dictval[str2][1])
              mat2 = eps.mul(std).add_(dictval[str2][0])
              mean1 = torch.mean(mat1, axis = 0)
              std1 = torch.std(mat1, axis = 0)
              mean2 = torch.mean(mat2, axis = 0)
              std2 = torch.std(mat2, axis = 0)
              vect = torch.cat([mean1, mean2, std1, std2], dim=0)
              self.samples[idx] = vect.detach().cpu()
              self.label[idx] = datat['label'][index]
              self.label[idx] = torch.tensor(self.label[idx], dtype=torch.int64)
              idx+=1
        self.sz = idx


    def __len__(self):
        return self.sz

    def __getitem__(self, idx):
        return self.samples[idx],self.label[idx]

class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size,n_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, n_classes)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x

class TestDataset(Dataset):
    def __init__(self,filename,dictval):
        datat = pd.read_csv(filename)
        #datat = datat.tail(tailpp).reset_index(drop=True)
       
        self.samples = list(range(len(datat)))
        self.label = list(range(len(datat)))
        self.sz = 0
        idx = 0
        for index in range(len(datat)) :
          str1 = datat['Drug1_ID'][index]
          str2 = datat['Drug2_ID'][index]

          
          if (str1 in dictval) and (str2 in dictval) :
            mat1 = dictval[str1][0]
            mat2 = dictval[str2][0]
            mean1 = torch.mean(mat1, axis = 0)
            std1 = torch.std(mat1, axis = 0)
            mean2 = torch.mean(mat2, axis = 0)
            std2 = torch.std(mat2, axis = 0)
            vect = torch.cat([mean1, mean2, std1, std2], dim=0)
            self.samples[idx] = vect.detach().cpu()
            self.label[idx] = datat['label'][index]
            self.label[idx] = torch.tensor(self.label[idx], dtype=torch.int64)
            idx+=1
        self.sz = idx


    def __len__(self):
        return self.sz

    def __getitem__(self, idx):
        return self.samples[idx],self.label[idx]

datatest = TestDataset("sup_test.csv")
len(datatest)