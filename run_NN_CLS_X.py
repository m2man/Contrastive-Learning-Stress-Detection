import torch
import numpy as np
import pandas as pd
import torch.nn as nn
device = torch.device('cuda:0')
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import itertools
from torch.utils.data import Dataset, DataLoader
import random
import math
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
import joblib

sigmoid_function = nn.Sigmoid()

EDA_FT = 35
EMB_EDA_SIZE = [35]
HR_FT = 25
EMB_HR_SIZE = [25]
CLS_SIZE = [64] # None or List

MAX_EPOCH = 50
OPTIM = 'ADAM'
LR = 0.003
DROPOUT = 0.6

MODEL_NAME = 'CLS_X_Adam'
SAVE_MODEL_DIR = 'Output_CLS'
NAME_DATASET = 'WESAD'
SUBJECT_ID_TEST = ['S9', 'S17', 'S6', 'S14', 'S3', 'S15', 'S11', 'S10', 'S13', 'S16', 'S2', 'S4', 'S5', 'S7', 'S8']

def logging(s, path, print_=False):
    if print_:
        print(s)
    if path:
        with open(path, 'a+') as f:
            f.write(s + '\n')
            
# Model
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
 
        self.emb = nn.Sequential(*modules)
        
    def forward(self, feat):
        x = self.emb(feat)
        #x = F.normalize(x, p=2, dim=1)
        return x
    
class Model_CLS(nn.Module):
    def __init__(self, input_eda_size, input_hr_size, emb_eda_size=None, emb_hr_size=None, dropout=0.2, cls_size=None):
        # cls_size, emb_eda_size, emb_hr_size is None or a list
        
        super(Model_CLS, self).__init__()
        self.dropout = dropout
        
        self.input_eda_size = input_eda_size
        if emb_eda_size is None:
            self.emb_eda_size = [input_eda_size]
        else:
            self.emb_eda_size = emb_eda_size
        self.eda_encode = Embedding_Model(input_size=self.input_eda_size, 
                                          dropout=self.dropout, 
                                          emb_size=self.emb_eda_size)
        
        self.input_hr_size = input_hr_size
        if emb_hr_size is None:
            self.emb_hr_size = [input_hr_size]
        else:
            self.emb_hr_size = emb_hr_size
        self.hr_encode = Embedding_Model(input_size=self.input_hr_size, 
                                         dropout=self.dropout, 
                                         emb_size=self.emb_hr_size)
        
        self.input_cls_size = self.emb_eda_size[-1] + self.emb_hr_size[-1] + input_hr_size + input_eda_size
        if cls_size is None:
            self.cls_size = [1]
        else:
            self.cls_size = cls_size + [1]
        self.cls = Embedding_Model(input_size=self.input_cls_size, 
                                   dropout=self.dropout, 
                                   emb_size=self.cls_size)
        
        self.eda_cls = Embedding_Model(input_size=self.emb_eda_size[-1]+input_hr_size, 
                                       dropout=self.dropout, 
                                       emb_size=[1])
        self.hr_cls = Embedding_Model(input_size=self.emb_hr_size[-1]+input_eda_size, 
                                       dropout=self.dropout, 
                                       emb_size=[1])
            
    def forward(self, eda_ft, hr_ft):
        eda_emb = self.eda_encode(eda_ft)
        hr_emb = self.hr_encode(hr_ft)
        
        eda_emb_cat = torch.cat((eda_emb, hr_ft), dim=1)
        hr_emb_cat = torch.cat((hr_emb, eda_ft), dim=1)
        
        eda_logit = self.eda_cls(eda_emb_cat)
        hr_logit = self.hr_cls(hr_emb_cat)
        
        com_emb = torch.cat((eda_emb, hr_emb, eda_ft, hr_ft), dim=1)
        com_logit = self.cls(com_emb)
        
        return eda_logit, hr_logit, com_logit
    
# Dataset
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
        hr_ft = sample_1[0:25]
        eda_ft = sample_1[25:]
        return eda_ft, hr_ft, label_1
        
    def __len__(self):
        return len(self.df)
    
def generate_batch_embed(batch):
    eda_fts, hr_fts,labels = zip(*batch)
    eda_ft = torch.tensor([ft for ft in eda_fts]).squeeze(1).float()#.to(device)
    hr_ft = torch.tensor([ft for ft in hr_fts]).squeeze(1).float()
    labels = torch.tensor(labels).float()#.to(device)
    return eda_ft, hr_ft, labels

# Dataloader Class
def make_EmbDataLoader(dataset, **args):
    data = DataLoader(dataset, collate_fn=generate_batch_embed, **args)
    return data


def train_epoch(model, optimizer, dataloader, loss_func):
    all_pred_eda = []
    all_pred_hr = []
    all_pred_com = []
    all_label = []
    eda_loss = []
    hr_loss = []
    com_loss = []
    total_loss = []
    dis_info = ''
    model.train()
        
    for i, (eda_ft, hr_ft, labels) in enumerate(dataloader):
        eda_ft = eda_ft.to(device)
        hr_ft = hr_ft.to(device)
        labels = labels.to(device)
        labels = labels.unsqueeze(-1)
        
        eda_logit, hr_logit, com_logit = model(eda_ft, hr_ft)
        
        loss_eda = loss_func(eda_logit, labels)
        loss_hr = loss_func(hr_logit, labels)
        loss_com = loss_func(com_logit, labels)
        loss_total = loss_eda + loss_hr + loss_com
        
        # Update Grad
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        
        with torch.no_grad():
            eda_cls = sigmoid_function(eda_logit)
            hr_cls = sigmoid_function(hr_logit)
            com_cls = sigmoid_function(com_logit)
       
            total_loss.append(loss_total.item())
            eda_loss.append(loss_eda.item())
            hr_loss.append(loss_hr.item())
            com_loss.append(loss_com.item())
            
            preds_eda = eda_cls.round().squeeze().detach().cpu().numpy().tolist()
            preds_hr = hr_cls.round().squeeze().detach().cpu().numpy().tolist()
            preds_com = com_cls.round().squeeze().detach().cpu().numpy().tolist()
            labels = labels.round().squeeze().detach().cpu().numpy().tolist()
            all_pred_eda += preds_eda
            all_pred_com += preds_com
            all_pred_hr += preds_hr
            all_label += labels

        str_info = f'[Iter {i}/{len(dataloader)}] Total: {round(np.mean(total_loss), 4)}\n'
        str_info += f'EDA Loss: {round(np.mean(eda_loss), 4)}\n'
        str_info += f'HR Loss: {round(np.mean(hr_loss), 4)}\n'
        str_info += f'COM Loss: {round(np.mean(com_loss), 4)}'
        if i % 100 == 0 or i == len(dataloader)-1:
            print(str_info)
            dis_info += str_info +'\n'

    dis_info += f"Class Samples Distribution: {len(all_label) - np.sum(all_label)} / {np.sum(all_label)}\n"
    
    bacc_eda = balanced_accuracy_score(all_label, all_pred_eda)
    acc_eda = accuracy_score(all_label, all_pred_eda)
    f1_eda = f1_score(all_label, all_pred_eda)    
    
    bacc_hr = balanced_accuracy_score(all_label, all_pred_hr)
    acc_hr = accuracy_score(all_label, all_pred_hr)
    f1_hr = f1_score(all_label, all_pred_hr)  
    
    bacc_com = balanced_accuracy_score(all_label, all_pred_com)
    acc_com = accuracy_score(all_label, all_pred_com)
    f1_com = f1_score(all_label, all_pred_com)  
    
    loss_dict = {}
    loss_dict['eda'] = np.mean(eda_loss)
    loss_dict['hr'] = np.mean(hr_loss)
    loss_dict['com'] = np.mean(com_loss)
    loss_dict['total'] = np.mean(total_loss)
    
    metric_dict = {}
    metric_dict['eda'] = {}
    metric_dict['eda']['acc'] = acc_eda
    metric_dict['eda']['bacc'] = bacc_eda
    metric_dict['eda']['f1'] = f1_eda
    metric_dict['hr'] = {}
    metric_dict['hr']['acc'] = acc_hr
    metric_dict['hr']['bacc'] = bacc_hr
    metric_dict['hr']['f1'] = f1_hr
    metric_dict['com'] = {}
    metric_dict['com']['acc'] = acc_com
    metric_dict['com']['bacc'] = bacc_com
    metric_dict['com']['f1'] = f1_com
    
    return dis_info, loss_dict, metric_dict

def validate_epoch(model, dataloader):
    all_pred_eda = []
    all_pred_hr = []
    all_pred_com = []
    all_label = []
    eda_loss = []
    hr_loss = []
    com_loss = []
    total_loss = []
    model.eval()
    
    with torch.no_grad():
        for i, (eda_ft, hr_ft, labels) in enumerate(dataloader):
            eda_ft = eda_ft.to(device)
            hr_ft = hr_ft.to(device)
            labels = labels.to(device)
            labels = labels.unsqueeze(-1)

            eda_logit, hr_logit, com_logit = model(eda_ft, hr_ft)

            eda_cls = sigmoid_function(eda_logit)
            hr_cls = sigmoid_function(hr_logit)
            com_cls = sigmoid_function(com_logit)
            
            preds_eda = eda_cls.round().squeeze().detach().cpu().numpy().tolist()
            preds_hr = hr_cls.round().squeeze().detach().cpu().numpy().tolist()
            preds_com = com_cls.round().squeeze().detach().cpu().numpy().tolist()
            labels = labels.round().squeeze().detach().cpu().numpy().tolist()
            all_pred_eda += preds_eda
            all_pred_com += preds_com
            all_pred_hr += preds_hr
            all_label += labels
    
    bacc_eda = balanced_accuracy_score(all_label, all_pred_eda)
    acc_eda = accuracy_score(all_label, all_pred_eda)
    f1_eda = f1_score(all_label, all_pred_eda)    
    
    bacc_hr = balanced_accuracy_score(all_label, all_pred_hr)
    acc_hr = accuracy_score(all_label, all_pred_hr)
    f1_hr = f1_score(all_label, all_pred_hr)  
    
    bacc_com = balanced_accuracy_score(all_label, all_pred_com)
    acc_com = accuracy_score(all_label, all_pred_com)
    f1_com = f1_score(all_label, all_pred_com)  
    
    metric_dict = {}
    metric_dict['eda'] = {}
    metric_dict['eda']['acc'] = acc_eda
    metric_dict['eda']['bacc'] = bacc_eda
    metric_dict['eda']['f1'] = f1_eda
    metric_dict['hr'] = {}
    metric_dict['hr']['acc'] = acc_hr
    metric_dict['hr']['bacc'] = bacc_hr
    metric_dict['hr']['f1'] = f1_hr
    metric_dict['com'] = {}
    metric_dict['com']['acc'] = acc_com
    metric_dict['com']['bacc'] = bacc_com
    metric_dict['com']['f1'] = f1_com
    
    return metric_dict

def metric_to_info(metric_dict):
    info = '~~~ EDA RESULT ~~~\n'
    info += f"Acc: {round(np.mean(metric_dict['eda']['acc']), 4)}"
    info += f" -- BAcc: {round(np.mean(metric_dict['eda']['bacc']), 4)}"
    info += f" -- F1: {round(np.mean(metric_dict['eda']['f1']), 4)}\n"
    info += '~~~ HR RESULT ~~~\n'
    info += f"Acc: {round(np.mean(metric_dict['hr']['acc']), 4)}"
    info += f" -- BAcc: {round(np.mean(metric_dict['hr']['bacc']), 4)}"
    info += f" -- F1: {round(np.mean(metric_dict['hr']['f1']), 4)}\n"
    info += '~~~ COM RESULT~~~\n'
    info += f"Acc: {round(np.mean(metric_dict['com']['acc']), 4)}"
    info += f" -- BAcc: {round(np.mean(metric_dict['com']['bacc']), 4)}"
    info += f" -- F1: {round(np.mean(metric_dict['com']['f1']), 4)}\n"
    return info

def loss_to_info(loss_dict):
    info = f"TOTAL Loss: {round(np.mean(loss_dict['total']), 4)}"
    info += f" -- EDA Loss: {round(np.mean(loss_dict['eda']), 4)}"
    info += f" -- HR Loss: {round(np.mean(loss_dict['hr']), 4)}"
    info += f" -- COM Loss: {round(np.mean(loss_dict['com']), 4)}"
    return info

##### READ DATASET #####
if NAME_DATASET == 'WESAD':
    DATA_DIR = '/home/nvtu/PhD_Work/StressDetection/DATA/MyDataset/WESAD'
    data_group = np.load(f'{DATA_DIR}/{NAME_DATASET}_WRIST_groups_1_60.npy')
    data_gt = np.load(f'{DATA_DIR}/{NAME_DATASET}_WRIST_ground_truth_1_60.npy')
    data_ft = np.load(f'{DATA_DIR}/{NAME_DATASET}_WRIST_stats_feats_1_60.npy')
else:
    DATA_DIR = '/home/nvtu/PhD_Work/StressDetection/DATA/MyDataset/AffectiveROAD_Data/Database'
    NAME_DATASET = 'AffectiveROAD'
    data_group = np.load(f'{DATA_DIR}/{NAME_DATASET}_groups_1.npy')
    data_gt = np.load(f'{DATA_DIR}/{NAME_DATASET}_ground_truth_1.npy')
    data_ft = np.load(f'{DATA_DIR}/{NAME_DATASET}_stats_feats_1.npy')
    indices = np.where(data_gt >= 0)[0]
    data_ft = data_ft[indices]
    data_group = data_group[indices]
    data_gt = data_gt[indices]

# Create dataframe for dataset
column_values = [f'f{x}' for x in range(data_ft.shape[1])]
data_full = pd.DataFrame(data = data_ft,  
                         columns = column_values)
data_full['subject_id'] = data_group
data_full['label'] = data_gt
list_subject_id = np.unique(data_full['subject_id']).tolist()

def main(subject_id_test):
    data_train_val = data_full[data_full.subject_id != subject_id_test]
    data_test = data_full[data_full.subject_id == subject_id_test]
    list_id = list(set(data_train_val.subject_id))
    list_id.sort()
    subject_id_validate = random.Random(1509+int(subject_id_test[1:])+88).choices(list_id,k=1)[0]
    if subject_id_test == 'S9':
        subject_id_validate = 'S8'
    data_train = data_train_val[data_train_val.subject_id != subject_id_validate]
    data_validate = data_train_val[data_train_val.subject_id == subject_id_validate]
    ft_names = data_train.columns.tolist()

    print(f"Training Model for {subject_id_test} and validate on {subject_id_validate}")

    #temp = Counter(data_train.iloc[:,-1].tolist())
    #print(temp)

    # Scaler Data
    X_train = data_train.iloc[:,:-2].to_numpy()
    group_train = data_train.iloc[:,-2].to_numpy()
    y_train = data_train.iloc[:,-1].to_numpy()
    X_test = data_test.iloc[:,:-2].to_numpy()
    group_test = data_test.iloc[:,-2].to_numpy()
    y_test = data_test.iloc[:,-1].to_numpy()
    X_validate = data_validate.iloc[:,:-2].to_numpy()
    group_validate = data_validate.iloc[:,-2].to_numpy()
    y_validate = data_validate.iloc[:,-1].to_numpy()

    X_train = X_train.astype('float64')
    X_validate = X_validate.astype('float64')
    X_test = X_test.astype('float64')

    y_train = y_train.astype('float64')
    y_validate = y_validate.astype('float64')
    y_test = y_test.astype('float64')

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_validate = scaler.transform(X_validate)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, f'{SAVE_MODEL_DIR}/{NAME_DATASET}/Model/StandardScaler_{subject_id_test}.joblib')

    # Create Dataframe
    df_train = pd.DataFrame(data = X_train, columns = ft_names[:-2])
    df_train['subject_id'] = group_train
    df_train['label'] = y_train

    df_validate = pd.DataFrame(data = X_validate, columns = ft_names[:-2])
    df_validate['subject_id'] = group_validate
    df_validate['label'] = y_validate

    df_test= pd.DataFrame(data = X_test, columns = ft_names[:-2])
    df_test['subject_id'] = group_test
    df_test['label'] = y_test

    train_dataset = EmbDataset(df_train)
    validate_dataset = EmbDataset(df_validate)
    test_dataset = EmbDataset(df_test)

    train_dataloader = make_EmbDataLoader(train_dataset, batch_size=512, shuffle=True)
    validate_dataloader = make_EmbDataLoader(validate_dataset, batch_size=2048, shuffle=False)
    test_dataloader = make_EmbDataLoader(test_dataset, batch_size=2048, shuffle=False)

    cls_func = nn.BCEWithLogitsLoss().to(device)

    Model = Model_CLS(input_eda_size=EDA_FT, input_hr_size=HR_FT, 
                      emb_eda_size=EMB_EDA_SIZE, emb_hr_size=EMB_HR_SIZE, 
                      dropout=DROPOUT, cls_size=CLS_SIZE)
    Model = Model.to(device)

    params = list(filter(lambda p: p.requires_grad, Model.parameters()))

    if OPTIM.lower() == 'adam':
        optimizer = torch.optim.Adam(params, lr=LR)
    if OPTIM.lower() == 'sgd':
        optimizer = torch.optim.SGD(params, lr=LR)
        
    ##### TRAIN #####
    min_loss = 100
    count_change_loss = 0
    scheduler = ReduceLROnPlateau(optimizer, factor = 0.2, patience=3, 
                                  mode = 'min', verbose=True, min_lr=1e-6)

    # Training
    for epoch in range(MAX_EPOCH):
        # train epoch
        train_info, loss_train_dict, metric_train_dict = train_epoch(Model, optimizer, train_dataloader, loss_func=cls_func)
        loss_train_info = loss_to_info(loss_train_dict)
        metric_train_info = metric_to_info(metric_train_dict)

        train_info = train_info + metric_train_info + '\n'

        metric_val_dict = validate_epoch(Model, validate_dataloader)
        validate_info = metric_to_info(metric_val_dict)
        val_info = f'[CLASSIFY VALIDATE {subject_id_validate} ONLY]\n'
        val_info += validate_info

        total_loss_val = 2 - metric_val_dict['com']['f1'] - metric_val_dict['com']['bacc']

        str_info = ''

        if total_loss_val < min_loss - 0.001:
            # save
            torch.save({'epoch': epoch, 
                        'model_state_dict': Model.state_dict()},
                        f"{SAVE_MODEL_DIR}/{NAME_DATASET}/Model/{MODEL_NAME}_{subject_id_test}.pth.tar")
            min_loss = total_loss_val
            str_info += f'[SAVE]'
            count_change_loss = 0
            print(str_info)
        else:
            count_change_loss += 1
        str_info += f'========== [{epoch}/{MAX_EPOCH-1}] ==========\n>>>>> TRAIN <<<<<\n'
        str_info += train_info + '>>>>> VAL <<<<<\n'
        str_info += val_info 

        print(f'========== [VALIDATE {epoch}/{MAX_EPOCH-1}] ==========\n' + val_info)

        scheduler.step(total_loss_val)

        logging(str_info+'\n', 
                f'{SAVE_MODEL_DIR}/{NAME_DATASET}/Log/Training_{MODEL_NAME}_{subject_id_test}.txt', 
                False)

        if count_change_loss >= 5:
            logging('Early Stopping', 
                    f'{SAVE_MODEL_DIR}/{NAME_DATASET}/Log/Training_{MODEL_NAME}_{subject_id_test}.txt', 
                    True)
            break

    #Test
    modelCheckpoint = torch.load(f"{SAVE_MODEL_DIR}/{NAME_DATASET}/Model/{MODEL_NAME}_{subject_id_test}.pth.tar")
    Model.load_state_dict(modelCheckpoint['model_state_dict'])
    Model.eval()

    print('Run on Test set in final --> LOAD Best Model')
    test_metric_dict = validate_epoch(Model, test_dataloader)
    test_metric_info = metric_to_info(test_metric_dict)
    test_info = f'[CLASSIFY TEST {subject_id_test} ONLY]\n'
    test_info += test_metric_info + '\n'
    logging(test_info, f'{SAVE_MODEL_DIR}/{NAME_DATASET}/Log/Training_{MODEL_NAME}_{subject_id_test}.txt', True)
    
for x in SUBJECT_ID_TEST:
    main(subject_id_test=x)