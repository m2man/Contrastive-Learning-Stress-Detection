import torch
import time
import itertools
import numpy as np
import torch.nn as nn
device = torch.device('cuda:0')
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

# An ordinary implementation of Swish function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# A memory-efficient implementation of Swish function
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)
    
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.constant_(m.bias, 0)
            
class Classify_Model(nn.Module):
    def __init__(self, input_size, dropout=0.4, hidden_size=[64]):
        super(Classify_Model, self).__init__()
        self.do = dropout
        self.input_size = input_size
        self.emb_size = hidden_size
        
#         modules = [] 
#         for idx, size in enumerate(self.emb_size):
#             if idx == 0:
#                 modules.append(nn.Linear(self.input_size, self.emb_size[idx]))
#             else:
#                 modules.append(nn.Linear(self.emb_size[idx-1], self.emb_size[idx]))
#             modules.append(nn.BatchNorm1d(num_features=self.emb_size[idx]))
#             modules.append(nn.ReLU())
#             modules.append(nn.Dropout(self.do))   
       
          # Add for Res connect
#         modules.append(nn.Linear(self.emb_size[-1], self.input_size))
#         modules.append(nn.BatchNorm1d(num_features=self.input_size))
#         modules.append(nn.ReLU())
#         modules.append(nn.Dropout(self.do))    
#         self.emb = nn.Sequential(*modules)
        
        self.cls = nn.Sequential(
            # --- #
            nn.Linear(self.input_size, 1),
            #nn.Sigmoid()
        )
        
        self.cls.apply(init_weights)

    def forward(self, feat):
        x = self.cls(feat)
        return x

class HeadProjection_Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(HeadProjection_Model, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        self.hd1 = nn.Linear(self.input_size, self.input_size)
        #self.bn1 = nn.BatchNorm1d(num_features=self.input_size)
        self.rl1 = nn.ReLU() #MemoryEfficientSwish() # nn.ReLU()
        #self.do = nn.Dropout(0.2)
        self.hd2 = nn.Linear(self.input_size, self.output_size)
        
    def forward(self, feat):
        x = self.hd1(feat)
        #x = self.bn1(x)
        x = self.rl1(x)
        #x = self.do(x)
        x = self.hd2(x)
        x = F.normalize(x, p=2, dim=1)
        return x    
    
class Embedding_Model(nn.Module):
    def __init__(self, input_size, dropout=0.4, emb_size=[64]):
        super(Embedding_Model, self).__init__()
        self.do = dropout
        self.input_size = input_size
        self.emb_size = emb_size
        
        modules = [] 
        for idx, size in enumerate(self.emb_size):
            if idx != len(self.emb_size) - 1: # Not the last layer
                if idx == 0:
                    modules.append(nn.Linear(self.input_size, self.emb_size[idx]))
                else:
                    modules.append(nn.Linear(self.emb_size[idx-1], self.emb_size[idx]))
                modules.append(nn.BatchNorm1d(num_features=self.emb_size[idx]))
                modules.append(nn.ReLU())
                modules.append(nn.Dropout(self.do))   
            else:
                # This is the last layer
                if idx == 0:
                    modules.append(nn.Linear(self.input_size, self.emb_size[idx]))
                else:
                    modules.append(nn.Linear(self.emb_size[idx-1], self.emb_size[idx]))
        # Add for Res connect
        #modules.append(nn.Linear(self.emb_size[-1], self.input_size))
        #modules.append(nn.BatchNorm1d(num_features=self.input_size))
        #modules.append(nn.ReLU())
        #modules.append(nn.Dropout(self.do))    
        self.emb = nn.Sequential(*modules)
        # self.emb.apply(init_weights)
        
    def forward(self, feat):
        x = self.emb(feat) #+ feat
        #x = F.normalize(x, p=2, dim=1)
        return x
    
class Sequence_Model(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=512, numb_layers=2, dropout=0.5, bidirectional=False, structure='GRU'):
        '''
        input_dim: int dim of input of sequence model (60)
        hidden_dim: int dim of hidden state of the sequence model (hidden state dim) (128)
        numb_layers: int number of sequence model (2)
        dropout: float dropout percent
        bidirectional: boolean apply bidirectional or not
        '''
        super(Sequence_Model, self).__init__()
        if structure == 'GRU':
            model = nn.GRU
        else:
            model = nn.LSTM
        self.model = model(input_size=input_dim, hidden_size=hidden_dim, num_layers=numb_layers, 
                           batch_first=True, dropout=dropout, bidirectional=bidirectional).to(device)
        
        #model = nn.GRU(input_size=2, hidden_size=10, num_layers=2, batch_first=True, bidirectional=True)
            
        self.numb_directions = 2 if bidirectional else 1
        self.numb_layers = numb_layers
        self.hidden_state = None
        self.hidden_dim = hidden_dim
        self.structure = structure
        
    def init_hidden(self, batch_size):
        if self.structure == 'GRU':
            return torch.zeros(self.numb_layers * self.numb_directions, batch_size, self.hidden_dim).to(device)
        else:
            return (torch.zeros(self.numb_layers * self.numb_directions, batch_size, self.hidden_dim).to(device),
                    torch.zeros(self.numb_layers * self.numb_directions, batch_size, self.hidden_dim).to(device))
        
    def forward(self, x, len_original_x):
        '''
        x is padded and embedded
        len_original_x is the len of un_padded of x
        '''
        batch_size, max_seq_len, input_dim = x.shape
        packed = pack_padded_sequence(x, len_original_x, batch_first=True, enforce_sorted=False)
        output, hidden = self.model(packed) # self.hidden
        out_unpacked, lens_out = pad_packed_sequence(output, batch_first=True)
        # out_unpacked is the embeded of each element in the seq
        if self.numb_directions == 2:
            out_unpacked_combine = (out_unpacked[:,:,:int(out_unpacked.shape[-1]/2)] + out_unpacked[:,:,int(out_unpacked.shape[-1]/2):])/2
        else:
            out_unpacked_combine = out_unpacked
            
        # Extract last hidden state
        if self.structure == 'GRU':
            final_state = hidden.view(self.numb_layers, self.numb_directions, batch_size, self.hidden_dim)[-1]
        else:
            final_state = hidden[0].view(self.numb_layers, self.numb_directions, batch_size, self.hidden_dim)[-1]
        
        # Handle directions
        if self.numb_directions == 1:
            final_hidden_state = final_state.squeeze(0)
        else:
            h_1, h_2 = final_state[0], final_state[1]
            final_hidden_state = (h_1 + h_2)/2               # Add both states (requires changes to the input size of first linear layer + attention layer)
            #final_hidden_state = torch.cat((h_1, h_2), 1)  # Concatenate both states
        
        return final_hidden_state#, out_unpacked_combine # (batch, hidden_size), (batch, max seq len, hidden_size)


class Model_Combine(nn.Module):
    def __init__(self, input_size, dropout=0.4, emb_size=[64], projection_size=64, cls_size=None):
        # cls_size is None or a list
        super(Model_Combine, self).__init__()
        self.encoder = Embedding_Model(input_size=input_size, dropout=dropout, emb_size=emb_size)
        self.projection_size = projection_size
        self.cls_size = cls_size
        if self.projection_size is not None:
            self.projection = HeadProjection_Model(input_size=emb_size[-1], output_size=projection_size)
        else:
            self.projection = nn.Identity()
        if self.cls_size is not None:
            self.cls = Classify_Model(emb_size[-1], dropout=dropout, hidden_size=cls_size)
        else:
            self.cls = None
            
    def forward(self, feat):
        encode = self.encoder(feat)
        if self.projection_size is not None:
            x = F.normalize(self.projection(encode), p=2, dim=1)
        else:
            x = self.projection(encode)
        if self.cls is not None:
            out = self.cls(encode)
        else:
            out = None
        return x, encode, out