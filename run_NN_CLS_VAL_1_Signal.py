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
EMB_EDA_SIZE = [70, 35]
HR_FT = 27
EMB_HR_SIZE = [54, 27]
CLS_SIZE = None # None or List

USE_FT = 'hr'
if USE_FT == 'eda':
    SIG_FT = EDA_FT
    EMB_SIG_SIZE = EMB_EDA_SIZE
else:
    SIG_FT = HR_FT
    EMB_SIG_SIZE = EMB_HR_SIZE

MAX_EPOCH = 50
USE_RES = False
OPTIM = 'Adam'
LR = 0.003
DROPOUT = 0.5
ACTIVATION = 'relu'
MODEL_NAME = 'Adam_Encode_Balance'
SAVE_MODEL_DIR = 'Output_General'
SUBFOLDER = f'{USE_FT.upper()}_DO5_120'
NAME_DATASET = 'WESAD'
if NAME_DATASET == 'WESAD':
    #SUBJECT_ID_TEST = ['S14', 'S9', 'S8', 'S11', 'S17', 'S13', 'S15', 'S3']
    #SUBJECT_ID_TEST += ['S6', 'S10', 'S16', 'S2', 'S4', 'S5', 'S7']
    SUBJECT_ID_TEST = ['S7']
    #SUBJECT_ID_TEST = ['S13']
else:
    #SUBJECT_ID_TEST = ['GM1', 'EK1', 'NM1', 'RY1', 'KSG1', 'AD1', 'NM3']
    #SUBJECT_ID_TEST = ['SJ1', 'BK1', 'RY2', 'GM2', 'MT1', 'NM2']
    #SUBJECT_ID_TEST = ['Drv1', 'Drv2', 'Drv3', 'Drv4', 'Drv5', 'Drv6', 'Drv7']
    SUBJECT_ID_TEST = ['Drv8', 'Drv9', 'Drv10', 'Drv11', 'Drv12', 'Drv13']

OPT = {}
OPT['learning_rate'] = LR
OPT['cosine'] = True
OPT['lr_decay_rate'] = 0.1
OPT['epochs'] = MAX_EPOCH
OPT['lr_decay_epochs'] = [5, 10, 20]
OPT['warm'] = True
OPT['warm_epochs'] = 10
OPT['warmup_from'] = 0.1
if OPT['cosine']:
    eta_min = OPT['learning_rate'] * (OPT['lr_decay_rate'] ** 3)
    OPT['warmup_to'] = eta_min + (OPT['learning_rate'] - eta_min) * (
            1 + math.cos(math.pi * OPT['warm_epochs'] / OPT['epochs'])) / 2
else:
    OPT['warmup_to'] = OPT['learning_rate']

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
    
def adjust_learning_rate(opt, optimizer, epoch): # begin of each epoch
    lr = opt['learning_rate']
    if opt['cosine']:
        eta_min = lr * (opt['lr_decay_rate'] ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / opt['epochs'])) / 2
    else:
        steps = np.sum(epoch > np.asarray(opt['lr_decay_epochs']))
        if steps > 0:
            lr = lr * (opt['lr_decay_rate'] ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(opt, epoch, batch_id, total_batches, optimizer): # in iteration in epoch
    if opt['warm'] and epoch <= opt['warm_epochs']:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (opt['warm_epochs'] * total_batches)
        lr = opt['warmup_from'] + p * (opt['warmup_to'] - opt['warmup_from'])

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
def logging(s, path, print_=False):
    if print_:
        print(s)
    if path:
        with open(path, 'a+') as f:
            f.write(s + '\n')
            
# Model
class Embedding_Model(nn.Module):
    def __init__(self, input_size, dropout=0.4, emb_size=[64], activation='relu'):
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
                if activation == 'relu':
                    modules.append(nn.ReLU())
                else:
                    modules.append(MemoryEfficientSwish())
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
    def __init__(self, input_signal_size, emb_signal_size=None, dropout=0.2, cls_size=None, activation='relu'):
        # cls_size, emb_eda_size, emb_hr_size is None or a list
        
        super(Model_CLS, self).__init__()
        self.dropout = dropout    
        self.input_signal_size = input_signal_size
        if emb_signal_size is None:
            self.emb_signal_size = [input_signal_size]
        else:
            self.emb_signal_size = emb_signal_size
        self.signal_encode = Embedding_Model(input_size=self.input_signal_size, 
                                             dropout=self.dropout, 
                                             emb_size=self.emb_signal_size, activation=activation)
        

        self.input_cls_size = self.emb_signal_size[-1]
        if cls_size is None:
            self.cls_size = [1]
        else:
            self.cls_size = cls_size + [1]
        self.cls = Embedding_Model(input_size=self.input_cls_size, 
                                   dropout=self.dropout, 
                                   emb_size=self.cls_size, activation=activation)
            
    def forward(self, signal_ft):
        signal_emb = self.signal_encode(signal_ft)
        signal_logit = self.cls(signal_emb)
        return signal_logit
    
# Dataset
class EmbDataset(Dataset):
    def __init__(self, df, signal='eda'):
        '''
        df is the dataframe containing features + subject_id + label
        numb_samples (int) total of pairs (samples) want to generated
        '''
        self.df = df.copy()
        self.signal = signal
        
    def __getitem__(self, i):
        sample_1 = self.df.iloc[i,:-2].to_numpy(dtype=np.float64) # not include subject_id and label
        label_1 = self.df['label'][i]
        hr_ft = sample_1[0:27]
        eda_ft = sample_1[27:]
        if self.signal == 'eda':
            return eda_ft, label_1
        else:
            return hr_ft, label_1
        
    def __len__(self):
        return len(self.df)
    
def generate_batch_embed(batch):
    signal_fts, labels = zip(*batch)
    signal_ft = torch.tensor([ft for ft in signal_fts]).squeeze(1).float()#.to(device)
    labels = torch.tensor(labels).float()#.to(device)
    return signal_ft, labels

# Dataloader Class
def make_EmbDataLoader(dataset, **args):
    data = DataLoader(dataset, collate_fn=generate_batch_embed, **args)
    return data


def train_epoch(model, optimizer, dataloader, loss_func):
    all_pred_signal = []
    all_label = []
    total_loss = []
    dis_info = ''
    model.train()
        
    for i, (signal_ft, labels) in enumerate(dataloader):
        signal_ft = signal_ft.to(device)
        labels = labels.to(device)
        labels = labels.unsqueeze(-1)
        signal_logit = model(signal_ft)
        loss_signal = loss_func(signal_logit, labels)
        
        # Update Grad
        optimizer.zero_grad()
        loss_signal.backward()
        optimizer.step()
        
        with torch.no_grad():
            signal_cls = sigmoid_function(signal_logit)
            total_loss.append(loss_signal.item())
            preds_signal = signal_cls.round().squeeze().detach().cpu().numpy().tolist()
            labels = labels.round().squeeze().detach().cpu().numpy().tolist()
            all_pred_signal += preds_signal
            all_label += labels

        str_info = f'[Iter {i}/{len(dataloader)}] Total: {round(np.mean(total_loss), 4)}'
        if i % 100 == 0 or i == len(dataloader)-1:
            print(str_info)
            dis_info += str_info +'\n'

    dis_info += f"Class Samples Distribution: {len(all_label) - np.sum(all_label)} / {np.sum(all_label)}\n"
    
    bacc_signal = balanced_accuracy_score(all_label, all_pred_signal)
    acc_signal = accuracy_score(all_label, all_pred_signal)
    f1_signal = f1_score(all_label, all_pred_signal)    
    
    loss_value = np.mean(total_loss)
    
    metric_dict = {}
    metric_dict['sig'] = {}
    metric_dict['sig']['acc'] = acc_signal
    metric_dict['sig']['bacc'] = bacc_signal
    metric_dict['sig']['f1'] = f1_signal
    
    return dis_info, loss_value, metric_dict

def validate_epoch(model, dataloader):
    all_pred_signal = []
    all_label = []
    total_loss = []
    model.eval()
    
    with torch.no_grad():
        for i, (signal_ft, labels) in enumerate(dataloader):
            signal_ft = signal_ft.to(device)
            labels = labels.to(device)
            labels = labels.unsqueeze(-1)
            signal_logit = model(signal_ft)
            signal_cls = sigmoid_function(signal_logit)
            preds_signal = signal_cls.round().squeeze().detach().cpu().numpy().tolist()
            labels = labels.round().squeeze().detach().cpu().numpy().tolist()
            all_pred_signal += preds_signal
            all_label += labels
    
    bacc_signal = balanced_accuracy_score(all_label, all_pred_signal)
    acc_signal = accuracy_score(all_label, all_pred_signal)
    f1_signal = f1_score(all_label, all_pred_signal)  
    
    metric_dict = {}
    metric_dict['sig'] = {}
    metric_dict['sig']['acc'] = acc_signal
    metric_dict['sig']['bacc'] = bacc_signal
    metric_dict['sig']['f1'] = f1_signal 
    return metric_dict

def test_model_dict(model_dict, dataloader_dict):
    list_validate_id = list(model_dict.keys())
    all_pred_dict = {}
    all_pred_prob_dict = {}
    
    with torch.no_grad():
        for idx_id, subject_id_validate in enumerate(list_validate_id):
            dataloader = dataloader_dict[subject_id_validate]
            model = model_dict[subject_id_validate]
            model.eval()
            
            all_pred = []
            all_pred_prob = []
    
            if idx_id == 0:
                all_label = []
                
            for i, (signal_ft, labels) in enumerate(dataloader):
                signal_ft = signal_ft.to(device)
                labels = labels.to(device)
                labels = labels.unsqueeze(-1)

                signal_logit = model(signal_ft)
                signal_cls = sigmoid_function(signal_logit)
            
                preds_round_list = signal_cls.round().squeeze().detach().cpu().numpy().tolist()
                preds_prob_list = signal_cls.squeeze().detach().cpu().numpy().tolist()
                labels = labels.round().squeeze().detach().cpu().numpy().tolist()
                
                all_pred += preds_round_list
                all_pred_prob += preds_prob_list
                
                if idx_id == 0:
                    all_label += labels
                    
            all_pred_dict[subject_id_validate] = all_pred
            all_pred_prob_dict[subject_id_validate] = all_pred_prob
    
    for idx, subject_id_validate in enumerate(list_validate_id):
        if idx == 0:
            f_pred = np.array([all_pred_dict[subject_id_validate]])
            f_pred_prob = np.array([all_pred_prob_dict[subject_id_validate]])
        else:
            f_pred = np.concatenate((f_pred, np.array([all_pred_dict[subject_id_validate]])), axis=0)
            f_pred_prob = np.concatenate((f_pred_prob, np.array([all_pred_prob_dict[subject_id_validate]])), axis=0)
    
    f_pred = np.mean(f_pred, axis=0)
    f_pred_prob = np.mean(f_pred_prob, axis=0)

    f_pred = np.round(f_pred)
    f_pred_prob = np.round(f_pred_prob)
    
    bacc = balanced_accuracy_score(all_label, f_pred)
    acc = accuracy_score(all_label, f_pred)
    f1 = f1_score(all_label, f_pred)    
    
    bacc_prob = balanced_accuracy_score(all_label, f_pred_prob)
    acc_prob = accuracy_score(all_label, f_pred_prob)
    f1_prob = f1_score(all_label, f_pred_prob)    
    
    metric_dict = {}
    metric_dict['sig'] = {}
    metric_dict['sig']['acc'] = acc
    metric_dict['sig']['bacc'] = bacc
    metric_dict['sig']['f1'] = f1
    metric_dict['sig_prob'] = {}
    metric_dict['sig_prob']['acc'] = acc_prob
    metric_dict['sig_prob']['bacc'] = bacc_prob
    metric_dict['sig_prob']['f1'] = f1_prob
    
    return metric_dict

def metric_to_info(metric_dict):
    info = ''
    list_keys = list(metric_dict.keys())
    for key in list_keys:
        info += f'~~~ {key.upper()} RESULT ~~~\n'
        info += f"Acc: {round(np.mean(metric_dict[key]['acc']), 4)}"
        info += f" -- BAcc: {round(np.mean(metric_dict[key]['bacc']), 4)}"
        info += f" -- F1: {round(np.mean(metric_dict[key]['f1']), 4)}\n"
    return info

def loss_to_info(loss_value):
    info = f"TOTAL Loss: {round(loss_value, 4)}"
    return info

##### READ DATASET #####
if NAME_DATASET == 'WESAD':
    DATA_DIR = '/home/nvtu/PhD_Work/StressDetection/DATA/MyDataset/WESAD'
    data_group = np.load(f'{DATA_DIR}/{NAME_DATASET}_WRIST_groups_0.25_120.npy')
    data_gt = np.load(f'{DATA_DIR}/{NAME_DATASET}_WRIST_ground_truth_0.25_120.npy')
    data_ft = np.load(f'{DATA_DIR}/{NAME_DATASET}_WRIST_stats_feats_0.25_120.npy')
else:
    DATA_DIR = '/home/nvtu/PhD_Work/StressDetection/DATA/MyDataset/AffectiveROAD_Data/Database'
    NAME_DATASET = 'AffectiveROAD'
    data_group = np.load(f'{DATA_DIR}/{NAME_DATASET}_groups_1_60.npy')
    data_gt = np.load(f'{DATA_DIR}/{NAME_DATASET}_ground_truth_1_60.npy')
    data_ft = np.load(f'{DATA_DIR}/{NAME_DATASET}_stats_feats_1_60.npy')
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
    print(len(data_test))
    list_id = list(set(data_train_val.subject_id))
    list_id.sort()
    
    for subject_id_validate in list_id:
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
        n_pos = np.sum(y_train)
        n_neg = len(y_train) - n_pos

        X_train = X_train.astype('float64')
        X_validate = X_validate.astype('float64')
        X_test = X_test.astype('float64')

        y_train = y_train.astype('float64')
        y_validate = y_validate.astype('float64')
        y_test = y_test.astype('float64')

        scaler = StandardScaler()
        #scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train)
        X_validate = scaler.transform(X_validate)
        X_test = scaler.transform(X_test)
        joblib.dump(scaler, f'{SAVE_MODEL_DIR}/{NAME_DATASET}/Model/{SUBFOLDER}/Scaler_{subject_id_test}_{subject_id_validate}.joblib')

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

        train_dataset = EmbDataset(df_train, signal=USE_FT)
        validate_dataset = EmbDataset(df_validate, signal=USE_FT)

        #train_dataloader = make_EmbDataLoader(train_dataset, batch_size=256, shuffle=True)
        validate_dataloader = make_EmbDataLoader(validate_dataset, batch_size=2048, shuffle=False)

        cls_func = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_neg/n_pos)).to(device)
        
        torch.manual_seed(1509)
        Model = Model_CLS(input_signal_size=SIG_FT,
                          emb_signal_size=EMB_SIG_SIZE,
                          dropout=DROPOUT, cls_size=CLS_SIZE, activation=ACTIVATION)
        Model = Model.to(device)

        params = list(filter(lambda p: p.requires_grad, Model.parameters()))

        if OPTIM.lower() == 'adam':
            optimizer = torch.optim.Adam(params, lr=LR)
        if OPTIM.lower() == 'sgd':
            optimizer = torch.optim.SGD(params, lr=LR)

        ##### TRAIN #####
        min_loss = 100
        count_change_loss = 0
        #scheduler = ReduceLROnPlateau(optimizer, factor = 0.2, patience=2, 
        #                              mode = 'min', verbose=True, min_lr=1e-6)

        # Training
        for epoch in range(MAX_EPOCH):
            # train epoch
            train_dataloader = make_EmbDataLoader(train_dataset, batch_size=2048, shuffle=True)
            train_info, loss_train, metric_train_dict = train_epoch(Model, optimizer, train_dataloader, loss_func=cls_func)
            loss_train_info = loss_to_info(loss_train)
            metric_train_info = metric_to_info(metric_train_dict)

            train_info = train_info + metric_train_info + '\n'

            metric_val_dict = validate_epoch(Model, validate_dataloader)
            print(metric_val_dict)
            validate_info = metric_to_info(metric_val_dict)
            val_info = f'[CLASSIFY VALIDATE {subject_id_validate} ONLY]\n'
            val_info += validate_info

            total_loss_val = 1 - metric_val_dict['sig']['bacc']

            str_info = ''

            if total_loss_val < min_loss - 0.001:
                # save
                torch.save({'epoch': epoch, 
                            'model_state_dict': Model.state_dict()},
                            f"{SAVE_MODEL_DIR}/{NAME_DATASET}/Model/{SUBFOLDER}/{MODEL_NAME}_{subject_id_test}_{subject_id_validate}.pth.tar")
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

            #scheduler.step(total_loss_val)

            logging(str_info+'\n', 
                    f'{SAVE_MODEL_DIR}/{NAME_DATASET}/Log/{SUBFOLDER}/{MODEL_NAME}_{subject_id_test}_{subject_id_validate}.txt', 
                    False)

            if count_change_loss >= 7:
                logging('Early Stopping', 
                        f'{SAVE_MODEL_DIR}/{NAME_DATASET}/Log/{SUBFOLDER}/{MODEL_NAME}_{subject_id_test}_{subject_id_validate}.txt', 
                        True)
                break

    #Test
    print('Run on Test set in final ...')
    X_test = data_test.iloc[:,:-2].to_numpy()
    group_test = data_test.iloc[:,-2].to_numpy()
    y_test = data_test.iloc[:,-1].to_numpy()
    model_dict = {}
    dataloader_dict = {}
    for subject_id_validate in list_id:
        # Scaler Data
        scaler = joblib.load(f'{SAVE_MODEL_DIR}/{NAME_DATASET}/Model/{SUBFOLDER}/Scaler_{subject_id_test}_{subject_id_validate}.joblib')
        X_test_scaler = scaler.transform(X_test)
        # Create Dataframe
        df_test= pd.DataFrame(data = X_test_scaler, columns = ft_names[:-2])
        df_test['subject_id'] = group_test
        df_test['label'] = y_test
        # Create Dataset and Dataloader
        test_dataset = EmbDataset(df_test, signal=USE_FT)
        test_dataloader = make_EmbDataLoader(test_dataset, batch_size=2048, shuffle=False)
        dataloader_dict[subject_id_validate] = test_dataloader
        # Load Model
        Model = Model_CLS(input_signal_size=SIG_FT,
                          emb_signal_size=EMB_SIG_SIZE,
                          dropout=DROPOUT, cls_size=CLS_SIZE, activation=ACTIVATION)
        Model = Model.to(device)
        
        modelCheckpoint = torch.load(f"{SAVE_MODEL_DIR}/{NAME_DATASET}/Model/{SUBFOLDER}/{MODEL_NAME}_{subject_id_test}_{subject_id_validate}.pth.tar")
        Model.load_state_dict(modelCheckpoint['model_state_dict'])
        model_dict[subject_id_validate] = Model
        Model.eval()
        # Predict
        test_metric_dict = validate_epoch(Model, test_dataloader)
        test_metric_info = metric_to_info(test_metric_dict)
        test_info = f'[CLASSIFY TEST {subject_id_test} - VALIDATE {subject_id_validate}]\n'
        test_info += test_metric_info
        print(test_info)
        logging(test_info + '\n', f'{SAVE_MODEL_DIR}/{NAME_DATASET}/Log/{SUBFOLDER}/Test_{MODEL_NAME}_{subject_id_test}.txt', True)
    
    validate_test_metric_dict = test_model_dict(model_dict, dataloader_dict)
    test_metric_info = metric_to_info(validate_test_metric_dict)
    test_info = f'[FINAL CLASSIFY TEST {subject_id_test}]\n'
    test_info += test_metric_info
    print(test_info)
    logging(test_info + '\n', f'{SAVE_MODEL_DIR}/{NAME_DATASET}/Log/{SUBFOLDER}/Test_{MODEL_NAME}_{subject_id_test}.txt', True)
    
for x in SUBJECT_ID_TEST:
    main(subject_id_test=x)