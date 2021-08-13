import torch
import time
import itertools
import numpy as np
import torch.nn as nn
device = torch.device('cuda:0')
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

def logging(s, path, print_=False):
    if print_:
        print(s)
    if path:
        with open(path, 'a+') as f:
            f.write(s + '\n')
            
# Dataset Class
class ContrastiveDataset(Dataset):
    # Generate image sgg dataset for the retrieval stage
    def __init__(self, df, numb_samples, k=7, run_full=False):
        '''
        df is the dataframe containing features + subject_id + label
        numb_samples (int) total of pairs (samples) want to generated
        '''
        self.df = df.copy()
        self.numb_samples = numb_samples
        self.k = k
        self.index_0 = np.where(self.df.label == 0)[0].tolist()
        self.index_1 = np.where(self.df.label == 1)[0].tolist()
        
    def __getitem__(self, i):
        samples = self.samples
        sample_1 = self.df.iloc[samples[i][0],:-2].to_numpy(dtype=np.float64) # not include subject_id and label
        sample_2 = self.df.iloc[samples[i][1],:-2].to_numpy(dtype=np.float64)
        label_1 = self.df['label'][samples[i][0]]
        label_2 = self.df['label'][samples[i][1]]
        '''
        if label_1 == label_2: # can be binary 0, 1 --> here just make more general
            if self.df['subject_id'][samples[i][0]] == self.df['subject_id'][samples[i][1]]:
                label = 1
            else:
                label = 1 #- (1 - self.is_validate) * random.random() / 10
        else:
            #label = (1 - self.is_validate) * random.random() / 10
            label = 0
        '''
        if label_1 == 0: # This time surely that label_1 == label_2 already
            label = 0
        else:
            label = 1
        return sample_1, sample_2, label
    
    def summary(self):
        result = []
        samples = self.samples
        for i in range(len(samples)):
            label_1 = self.df['label'][samples[i][0]]
            label_2 = self.df['label'][samples[i][1]]
            subject_1 = self.df['subject_id'][samples[i][0]]
            subject_2 = self.df['subject_id'][samples[i][1]]
            if label_1 == label_2:
                if label_1 == 1:
                    if subject_1 != subject_2:
                        result += [4] # same label 1 but dif subj
                    else:
                        result += [3] # same label 1 same subj
                else:
                    if subject_1 != subject_2:
                        result += [2] # same label 0 but dif subj
                    else:
                        result += [1] # same label 0 same subj
            else:
                result += [0] # dif label
        return result
            
    def shuffle(self, seed=1509):
        index_0 = self.index_0.copy()
        index_1 = self.index_1.copy()
        random.Random(seed).shuffle(index_0)
        random.Random(seed).shuffle(index_1)
        index_0 = index_0[:min(len(index_0), 1000)]
        index_1 = index_1[:min(len(index_1), 1000)]
        print('Shuffling Dataset ...')
        pairs_0 = list(itertools.product(index_0,index_0))
        pairs_1 = list(itertools.product(index_1,index_1))
        pairs_all = pairs_0 + pairs_1
        #print(f'Total {len(pairs_all)} samples')
        # self.all_pairs = list(itertools.product(range(len(self.df)), repeat=2)) # [(0,0), (0,1), ...]
        self.weights = [self.k*(1-0.25*(self.df['subject_id'][pair[0]] != self.df['subject_id'][pair[1]])) \
                        if self.df['label'][pair[0]] == self.df['label'][pair[1]] and self.df['label'][pair[0]] == 1\
                        else self.k/10*(1+0.15*(self.df['subject_id'][pair[0]] != self.df['subject_id'][pair[1]])) \
                        if self.df['label'][pair[0]] == self.df['label'][pair[1]] and self.df['label'][pair[0]] == 0\
                        else 0 for pair in pairs_all]
        self.samples = random.choices(pairs_all, weights=self.weights, k=self.numb_samples)
        
    def __len__(self):
        return len(self.samples)
    
def generate_batch_contrastive(batch):
    samples_1, samples_2, labels = zip(*batch)
    ft1 = torch.tensor([sample for sample in samples_1]).squeeze(1).float()#.to(device)
    ft2 = torch.tensor([sample for sample in samples_2]).squeeze(1).float()#.to(device)
    labels = torch.tensor(labels).float()#.to(device)
    return ft1, ft2, labels

# Dataloader Class
def make_ContrastiveDataLoader(dataset, **args):
    data = DataLoader(dataset, collate_fn=generate_batch_contrastive, **args)
    return data

# Contrastive Model
class Contrastive_Model(nn.Module):
    def __init__(self, input_size, dropout=0.4, emb_size=[64]):
        super(Contrastive_Model, self).__init__()
        self.do = dropout
        self.input_size = input_size
        self.emb_size = emb_size
        
        modules = [] 
        for idx, size in enumerate(self.emb_size):
            if idx == 0:
                modules.append(nn.Linear(self.input_size, self.emb_size[idx]))
            else:
                modules.append(nn.Linear(self.emb_size[idx-1], self.emb_size[idx]))
            modules.append(nn.BatchNorm1d(num_features=self.emb_size[idx]))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(self.do))   
       
        # Add for Res connect
        modules.append(nn.Linear(self.emb_size[-1], self.input_size))
        modules.append(nn.BatchNorm1d(num_features=self.input_size))
        modules.append(nn.ReLU())
        modules.append(nn.Dropout(self.do))    
        self.emb = nn.Sequential(*modules)

    def forward(self, feat):
        x = self.emb(feat) + feat
        return x
    
# Loss function    
def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

def CosineSimilarity(images_geb, captions_geb):
    similarities = sim_matrix(images_geb, captions_geb) # n_img, n_caption
    return similarities

class ContrastiveLoss_CosineSimilarity(nn.Module):
    def __init__(self, margin=0, max_violation=False):
        super(ContrastiveLoss_CosineSimilarity, self).__init__()
        self.max_violation = max_violation
        self.margin = margin
        
    def forward(self, images_geb, captions_geb):
        scores = CosineSimilarity(images_geb, captions_geb)
        diagonal = scores.diag().view(len(images_geb), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.mean() + cost_im.mean()
    
def train_epoch_contrastive(model_con, optimizer, dataloader, loss_func_con):
    '''
    Train 1 epoch
    '''
    total_loss = []
    model_con.train()
    for i, (samples_1, samples_2, labels) in enumerate(dataloader):
        samples_1 = samples_1.to(device)
        samples_2 = samples_2.to(device)
        #labels = labels.to(device)
        
        optimizer.zero_grad()
        # Contrastive Model (Embedding)
        samples_1_emb = model_con(samples_1)
        samples_2_emb = model_con(samples_2)
        
        loss_total = loss_func_con(samples_1_emb, samples_2_emb)
        
        # Update Grad
        loss_total.backward()
        optimizer.step()
        
        total_loss.append(loss_total.item())
        
        str_info = f'[Iter {i}/{len(dataloader)}] Total: {round(np.mean(total_loss), 4)}'
        if i % 250 == 0:
            print(str_info)
            # logging(str_info, 'log.txt')
    print('Done Epoch')
    return model_con, optimizer, np.mean(total_loss)

def validate_contrastive(model_con, dataloader, loss_func_con):
    '''
    Validate model
    '''
    total_loss = []
    with torch.no_grad():
        model_con.eval()
        for i, (samples_1, samples_2, labels) in enumerate(dataloader):
            samples_1 = samples_1.to(device)
            samples_2 = samples_2.to(device)
            labels = labels.to(device)
            
            samples_1_emb = model_con(samples_1)
            samples_2_emb = model_con(samples_2)
            
            loss_total = loss_func_con(samples_1_emb, samples_2_emb)
        
            total_loss.append(loss_total.item())
                        
        total_loss = np.mean(total_loss)
    return total_loss

########## JUST TESTING CONTRASTIVE LOSS ############
def calculate_contrastive_loss(dataloader, loss_func_con):
    total_loss = []
    with torch.no_grad():
        for i, (samples_1, samples_2, labels) in enumerate(dataloader):
            samples_1 = samples_1.to(device)
            samples_2 = samples_2.to(device)
            labels = labels.to(device)
           
            loss_total = loss_func_con(samples_1, samples_2)
        
            total_loss.append(loss_total.item())
                        
        total_loss = np.mean(total_loss)
    return total_loss
#####################################################

class EmbedDataset(Dataset):
    def __init__(self, df):
        '''
        df is the dataframe containing features + subject_id + label
        numb_samples (int) total of pairs (samples) want to generated
        '''
        self.df = df.copy()
        
    def __getitem__(self, i):
        sample_1 = self.df.iloc[i,:-2].to_numpy(dtype=np.float64) # not include subject_id and label
        label_1 = self.df['label'][i]
        return sample_1, label_1
        
    def __len__(self):
        return len(self.df)
    
def generate_batch_embed(batch):
    samples ,labels = zip(*batch)
    ft = torch.tensor([sample for sample in samples]).squeeze(1).float()#.to(device)
    labels = torch.tensor(labels).float()#.to(device)
    return ft, labels

# Dataloader Class
def make_EmbedDataLoader(dataset, **args):
    data = DataLoader(dataset, collate_fn=generate_batch_embed, **args)
    return data

def embed_df(model_con, dataloader):
    all_ft_embed = []
    all_label = []
    with torch.no_grad():
        model_con.eval()
        for i, (samples, labels) in enumerate(dataloader):
            samples = samples.to(device)
            
            samples_emb = model_con(samples)
            samples_emb = samples_emb.cpu()
            all_ft_embed.append(samples_emb)
            all_label.append(labels)
    all_ft_embed = torch.cat(all_ft_embed)
    all_label = torch.cat(all_label)
    return all_ft_embed, all_label