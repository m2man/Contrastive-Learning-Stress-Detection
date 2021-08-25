import torch
import time
import itertools
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import math

##########################################3
class SeqConIntDataset(Dataset):
    # Generate image sgg dataset for the retrieval stage
    def __init__(self, df, look_before=2):
        '''
        df is the dataframe containing features + subject_id + label
        numb_samples (int) total of pairs (samples) want to generated
        '''
        self.df = df.copy()
        self.look_before = look_before
        self.seed = 1509
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, i):
        # 1, 2 is same subject and same label
        # 1, 3 is same subject but dif label
        idx_sample_1 = i
        sample_1 = self.df.iloc[idx_sample_1,:-2].to_numpy(dtype=np.float64) # not include subject_id and label
        label_1 = self.df['label'][idx_sample_1]
        subject_1 = self.df['subject_id'][idx_sample_1]
        
        same_w_1 = np.where((self.df['subject_id']==subject_1) & (self.df['label']==label_1))[0]
        same_w_1 = same_w_1.tolist()
        dif_w_1 = np.where((self.df['subject_id']==subject_1) & (self.df['label']!=label_1))[0]
        dif_w_1 = dif_w_1.tolist()
        
        idx_sample_2 = random.choice(same_w_1)
        idx_sample_3 = random.choice(dif_w_1)
        
        sample_2 = self.df.iloc[idx_sample_2,:-2].to_numpy(dtype=np.float64) # not include subject_id and label
        label_2 = self.df['label'][idx_sample_2]
        sample_3 = self.df.iloc[idx_sample_3,:-2].to_numpy(dtype=np.float64) # not include subject_id and label
        label_3 = self.df['label'][idx_sample_3]
        
        sample_1 = sample_1.reshape(1, len(sample_1))
        sample_2 = sample_2.reshape(1, len(sample_2))
        sample_3 = sample_3.reshape(1, len(sample_3))
        
        len_sample_1 = self.look_before + 1
        len_sample_2 = self.look_before + 1
        len_sample_3 = self.look_before + 1
        
        for idx in range(self.look_before):
            idx1 = idx_sample_1 - idx - 1
            if idx1 < 0:
                len_sample_1 = len_sample_1 - 1
                bs1 = np.zeros(sample_1.shape[-1], dtype=np.float64)
            else:
                bs1 =  self.df.iloc[idx1,:-2].to_numpy(dtype=np.float64)
            sample_1 = np.concatenate((bs1.reshape(1, len(bs1)), sample_1))
            
            idx2 = idx_sample_2 - idx - 1
            if idx2 < 0:
                len_sample_2 = len_sample_2 - 1
                bs2 = np.zeros(sample_2.shape[-1], dtype=np.float64)
            else:
                bs2 =  self.df.iloc[idx2,:-2].to_numpy(dtype=np.float64)
            sample_2 = np.concatenate((bs2.reshape(1, len(bs2)), sample_2))
            
            idx3 = idx_sample_3 - idx - 1
            if idx3 < 0:
                len_sample_3 = len_sample_3 - 1
                bs3 = np.zeros(sample_3.shape[-1], dtype=np.float64)
            else:
                bs3 =  self.df.iloc[idx3,:-2].to_numpy(dtype=np.float64)
            sample_3 = np.concatenate((bs3.reshape(1, len(bs3)), sample_3))
            
        return sample_1, sample_2, sample_3, len_sample_1, len_sample_2, len_sample_3, label_1, label_2, label_3
    
def generate_batch_1(batch):
    sample_1, sample_2, sample_3, len_sample_1, len_sample_2, len_sample_3, label_1, label_2, label_3 = zip(*batch)
    ft1 = torch.tensor([sample for sample in sample_1]).squeeze(1).float()#.to(device)
    ft2 = torch.tensor([sample for sample in sample_2]).squeeze(1).float()#.to(device)
    ft3 = torch.tensor([sample for sample in sample_3]).squeeze(1).float()#.to(device)
    label_1 = torch.tensor(label_1).float()#.to(device)
    label_2 = torch.tensor(label_2).float()#.to(device)
    label_3 = torch.tensor(label_3).float()#.to(device)
    len_sample_1 = torch.tensor(len_sample_1, dtype=torch.int32)#.to(device)
    len_sample_2 = torch.tensor(len_sample_2, dtype=torch.int32)
    len_sample_3 = torch.tensor(len_sample_3, dtype=torch.int32)
    return ft1, ft2, ft3, len_sample_1, len_sample_2, len_sample_3, label_1, label_2, label_3

# Dataloader Class
def make_SeqConIntDataLoader(dataset, **args):
    data = DataLoader(dataset, collate_fn=generate_batch_1, **args)
    return data
##########################################3



##########################################3
class SeqDataset(Dataset):
    def __init__(self, df, look_before=2):
        '''
        df is the dataframe containing features + subject_id + label
        numb_samples (int) total of pairs (samples) want to generated
        '''
        self.df = df.copy()
        self.look_before = look_before
        
    def __getitem__(self, i):
        sample_1 = self.df.iloc[i,:-2].to_numpy(dtype=np.float64) # not include subject_id and label
        label_1 = self.df['label'][i]
        sample_1 = sample_1.reshape(1, len(sample_1))
        len_sample_1 = self.look_before + 1
        
        for idx in range(self.look_before):
            idx1 = i - idx - 1
            if idx1 < 0:
                len_sample_1 = len_sample_1 - 1
                bs1 = np.zeros(sample_1.shape[-1], dtype=np.float64)
            else:
                bs1 =  self.df.iloc[idx1,:-2].to_numpy(dtype=np.float64)
            sample_1 = np.concatenate((bs1.reshape(1, len(bs1)), sample_1))
        return sample_1, len_sample_1, label_1
        
    def __len__(self):
        return len(self.df)
    
def generate_batch_2(batch):
    samples_1, len_samples_1, labels = zip(*batch)
    ft1 = torch.tensor([sample for sample in samples_1]).squeeze(1).float()#.to(device)
    labels = torch.tensor(labels).float()#.to(device)
    len_samples_1 = torch.tensor(len_samples_1, dtype=torch.int32)#.to(device)
    return ft1, len_samples_1, labels

# Dataloader Class
def make_SeqDataLoader(dataset, **args):
    data = DataLoader(dataset, collate_fn=generate_batch_2, **args)
    return data
##########################################




##########################################
class EmbConDataset(Dataset):
    # Generate image sgg dataset for the retrieval stage
    def __init__(self, df, numb_samples, k=7, internal_each_sample=200):
        '''
        df is the dataframe containing features + subject_id + label
        numb_samples (int) total of pairs (samples) want to generated
        '''
        self.df = df.copy()
        self.numb_samples = numb_samples
        self.k = k
        self.index_0 = np.where(self.df.label == 0)[0].tolist()
        self.index_1 = np.where(self.df.label == 1)[0].tolist()
        
        self.internal_each_sample = internal_each_sample
        self.list_subject = list(set(self.df['subject_id']))
        self.index_dict = {}
        self.flag = {}
        for subj in self.list_subject:
            self.index_dict[subj] = {}
            index_0 = np.where((self.df.label == 0) & (self.df.subject_id == subj))[0].tolist()
            index_1 = np.where((self.df.label == 1) & (self.df.subject_id == subj))[0].tolist()
            random.Random(internal_each_sample).shuffle(index_0)
            random.Random(internal_each_sample).shuffle(index_1)
            self.index_dict[subj][0] = index_0
            self.index_dict[subj][1] = index_1
            self.flag[subj] = {}
            self.flag[subj][0] = 0
            self.flag[subj][1] = 1
            
        self.all_is_process = False
        self.calibrate_classes(seed=1509, reverse=True)
        self.batch_0 = 60
        self.batch_1 = 60
        self.start_0 = 0
        self.start_1 = 0
    
    def calibrate_classes(self, seed=1509, reverse=False):
        # Concate sample 1 to have same length of sample 0
        self.index_0_sample_all = self.index_0.copy()
        self.index_1_sample_all = self.index_1.copy()
        random.Random(seed).shuffle(self.index_0_sample_all)
        random.Random(seed).shuffle(self.index_1_sample_all)
        self.start_0 = 0
        self.start_1 = 0
        if reverse:
            self.index_0_sample_all = self.index_0_sample_all[:len(self.index_1_sample_all)]
        else:
            self.portion_0_over_1 = math.ceil(len(self.index_0_sample_all)/len(self.index_1_sample_all))
            self.index_1_sample_all = [self.index_1_sample_all for x in range(self.portion_0_over_1)]
            self.index_1_sample_all = list(itertools.chain.from_iterable(self.index_1_sample_all))
            self.index_1_sample_all = self.index_1_sample_all[:len(self.index_0_sample_all)]
        
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
        pairs_0 = list(itertools.combinations(index_0,r=2))
        pairs_1 = list(itertools.combinations(index_1,r=2))
        pairs_all = pairs_0 + pairs_1
        #print(f'Total {len(pairs_all)} samples')
        # self.all_pairs = list(itertools.product(range(len(self.df)), repeat=2)) # [(0,0), (0,1), ...]
        self.weights = [self.k*(1-0.25*(self.df['subject_id'][pair[0]] != self.df['subject_id'][pair[1]])) \
                        if self.df['label'][pair[0]] == self.df['label'][pair[1]] and self.df['label'][pair[0]] == 1\
                        else self.k*(1-0.15*(self.df['subject_id'][pair[0]] != self.df['subject_id'][pair[1]])) \
                        if self.df['label'][pair[0]] == self.df['label'][pair[1]] and self.df['label'][pair[0]] == 0\
                        else 0 for pair in pairs_all]
        self.samples = random.Random(seed).choices(pairs_all, weights=self.weights, k=self.numb_samples)
#         dis = self.summary()
#         dis_counter = Counter(dis)
#         numb_same = int((dis_counter[1] + dis_counter[3])/2)
#         dis = np.asarray(dis)
#         idx_2 = np.where(dis == 2)[0].tolist()
#         idx_4 = np.where(dis == 4)[0].tolist()
#         idx_1 = np.where(dis == 1)[0].tolist()
#         idx_3 = np.where(dis == 3)[0].tolist()
#         random.Random(seed).shuffle(idx_2)
#         random.Random(seed).shuffle(idx_4)
#         idx_2 = idx_2[:numb_same]
#         idx_4 = idx_4[:numb_same]
#         idx_all = idx_1 + idx_2 + idx_3 + idx_4
#         self.samples = np.asarray(self.samples)[idx_all].tolist()
    
    def shuffle_internal(self, seed=1509):
        print('Shuffling Dataset Internal ...')
        pairs_internal = []
        for idx, subj in enumerate(self.list_subject):
            idx_0 = self.index_dict[subj][0].copy()
            idx_1 = self.index_dict[subj][1].copy()
            random.Random(seed+idx).shuffle(idx_0)
            random.Random(seed+idx).shuffle(idx_1)
            idx_0 = idx_0[:self.internal_each_sample]
            idx_1 = idx_1[:self.internal_each_sample]
            pairs_0 = list(itertools.combinations(idx_0,r=2))
            pairs_1 = list(itertools.combinations(idx_1,r=2))
            pairs_internal = pairs_internal + pairs_0 + pairs_1
            
        random.Random(seed).shuffle(pairs_internal)
        print(f'Number of Internal Pairs {len(pairs_internal)}')
        self.samples = pairs_internal        
        
    def sampling_all_internal(self, seed=1509, cross=False):
        print('Shuffling Dataset Internal ...')
        pairs_internal = []
        idx_subj = {}
        for idx, subj in enumerate(self.list_subject):
            idx_0 = self.index_dict[subj][0].copy()
            idx_1 = self.index_dict[subj][1].copy()
            
            start_0 = self.flag[subj][0]*self.internal_each_sample
            end_0 = min(len(idx_0), start_0 + self.internal_each_sample)
            idx_0 = idx_0[start_0:end_0]
            
            start_1 = self.flag[subj][1]*self.internal_each_sample 
            end_1 = min(len(idx_1), start_1 + self.internal_each_sample)
            idx_1 = idx_1[start_1:end_1]
            
            idx_subj[subj]={}
            idx_subj[subj][0] = idx_0
            idx_subj[subj][1] = idx_1
            
            pairs_0 = list(itertools.combinations(idx_0,r=2))
            pairs_1 = list(itertools.combinations(idx_1,r=2))
            pairs_internal = pairs_internal + pairs_0 + pairs_1
            
            if (self.flag[subj][0]+1)*self.internal_each_sample >= len(self.index_dict[subj][0]):   
                self.flag[subj][0] = 0
                print(f"Roll out {subj} class 0")
            else:
                self.flag[subj][0] += 1
                
            if (self.flag[subj][1]+1)*self.internal_each_sample >= len(self.index_dict[subj][1]):   
                self.flag[subj][1] = 0
                print(f"Roll out {subj} class 1")
            else:
                self.flag[subj][1] += 1
        
        # Cross subject
        if cross:
            for idx, subj in enumerate(self.list_subject):
                idx_0_current_subj = idx_subj[subj][0]
                idx_1_current_subj = idx_subj[subj][1]
                for next_idx in range(idx+1, len(self.list_subject)):
                    next_subj = self.list_subject[next_idx]
                    idx_0_next_subj = idx_subj[next_subj][0]
                    idx_1_next_subj = idx_subj[next_subj][1]
                    pairs_0 = list(itertools.product(idx_0_current_subj, idx_0_next_subj))
                    pairs_1 = list(itertools.product(idx_1_current_subj, idx_1_next_subj))
                    pairs_internal = pairs_internal + pairs_0 + pairs_1
                
        random.Random(seed).shuffle(pairs_internal)
        print(f'Number of Internal Pairs {len(pairs_internal)}')
        self.samples = pairs_internal        
    
    def sampling_all(self, seed=1509):
        if self.all_is_process:
            self.all_is_process = False
            self.calibrate_classes(seed=seed, reverse=True)
        
        pairs_0 = []
        for val in range(self.start_0, self.start_0 + self.batch_0):
            anchor = self.index_0_sample_all[val]
            list_match = self.index_0_sample_all[(val+1):]
            p = list(itertools.product([anchor], list_match))
            pairs_0 += p 
            if (val + 1) >= len(self.index_0_sample_all):
                print('Roll out sample 0')
                self.all_is_process = True
                break
                
        pairs_1 = []
        for val in range(self.start_1, self.start_1 + self.batch_1):
            anchor = self.index_1_sample_all[val]
            list_match = self.index_1_sample_all[(val+1):]
            p = list(itertools.product([anchor], list_match))
            pairs_1 += p 
            if (val + 1) >= len(self.index_1_sample_all):
                print('Roll out sample 1')
                break
                
        self.start_0 = self.start_0 + self.batch_0
        self.start_1 = self.start_1 + self.batch_1
        
        pairs_all = pairs_0 + pairs_1
        random.Random(seed).shuffle(pairs_all)
        print(f'Number of Pairs {len(pairs_all)}')
        self.samples = pairs_all        
        
    def __len__(self):
        return len(self.samples)
    
def generate_batch_3(batch):
    samples_1, samples_2, labels = zip(*batch)
    ft1 = torch.tensor([sample for sample in samples_1]).squeeze(1).float()#.to(device)
    ft2 = torch.tensor([sample for sample in samples_2]).squeeze(1).float()#.to(device)
    labels = torch.tensor(labels).float()#.to(device)
    return ft1, ft2, labels

# Dataloader Class
def make_EmbConDataLoader(dataset, **args):
    data = DataLoader(dataset, collate_fn=generate_batch_3, **args)
    return data
##########################################




##########################################
class EmbDataset(Dataset):
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
def make_EmbDataLoader(dataset, **args):
    data = DataLoader(dataset, collate_fn=generate_batch_embed, **args)
    return data