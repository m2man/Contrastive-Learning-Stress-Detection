import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, RobustScaler
from collections import Counter
from Model_Utils import Sequence_Model, Classify_Model, Embedding_Model, HeadProjection_Model, Model_Combine
from Dataset_Utils import *
from General_Utils import *
from Loss_Utils import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
import joblib
from scipy import stats
from torchlars import LARS

# 14, 6, 17, 3, 2, 13, 9, 10, 15, 8, 7, 11, 4, 5, 16
SUBJECT_ID_TEST = [6, 17, 3, 2, 13, 9, 10, 15, 8, 7, 11, 4, 5]
SUBJECT_ID_TEST = ['S'+str(x) for x in SUBJECT_ID_TEST]
SUBJECT_ID_TEST = 'S9'

# # 'GM1', 'EK1', 'NM1', 'RY1', 'KSG1', 'AD1', 'NM3', 'SJ1', 'BK1', 'RY2', 'GM2', 'MT1', 'NM2'
# SUBJECT_ID_TEST = 'AD1' # SJ1

USE_LARS = True
OPTIM = 'Adam'
LR = 0.003

MARGIN = 1
SEQ_DIM = 128
INPUT_FT = 60
EMBEDDING_HIDDEN = [256, 512]
PROJECTION_OUT = None #128 # None or int
CLASSIFY_HIDDEN = [256] # [64] # list or None
DROPOUT = 0.2

MAX_EPOCH = 50
LOOK_BEFORE = 3
INTERNAL_SAMPLE = 200

MODEL_NAME = 'SCL_CLS_512_LARS_Adam'
SAVE_MODEL_DIR = 'Output'
NAME_DATASET = 'WESAD'

##### READ DATASET #####
if NAME_DATASET == 'WESAD':
    DATA_DIR = '/home/nvtu/PhD_Work/StressDetection/DATA/MyDataset/WESAD'
    data_group = np.load(f'{DATA_DIR}/{NAME_DATASET}_WRIST_groups_1.npy')
    data_gt = np.load(f'{DATA_DIR}/{NAME_DATASET}_WRIST_ground_truth_1.npy')
    data_ft = np.load(f'{DATA_DIR}/{NAME_DATASET}_WRIST_stats_feats_1.npy')
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

def train_model(subject_id_test='S10', pretrained_model=''):
    ##### TRAIN / VAL / TEST #####
    data_train_val = data_full[data_full.subject_id != subject_id_test]
    data_test = data_full[data_full.subject_id == subject_id_test]
    list_id = list(set(data_train_val.subject_id))
    list_id.sort()
    subject_id_validate = random.Random(1509+int(subject_id_test[1:])+88).choices(list_id,k=1)[0]
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

    ##### DATASET / DATALOADER #####
#     train_dataset = SeqConIntDataset(df=df_train, look_before=LOOK_BEFORE)
#     validate_dataset = SeqConIntDataset(df=df_validate, look_before=LOOK_BEFORE)
#     test_dataset = SeqConIntDataset(df=df_test, look_before=LOOK_BEFORE)
    
#     train_loader = make_SeqConIntDataLoader(train_dataset, batch_size=1024, shuffle=True)
#     validate_loader = make_SeqConIntDataLoader(validate_dataset, batch_size=1024, shuffle=False)
#     test_loader = make_SeqConIntDataLoader(test_dataset, batch_size=1024, shuffle=False)
        
#     validate_dataset_embed = SeqDataset(df_validate, look_before=LOOK_BEFORE)
#     validate_dataloader_embed = make_SeqDataLoader(validate_dataset_embed, batch_size=1024, shuffle=False)
    
#     test_dataset_embed = SeqDataset(df_test, look_before=LOOK_BEFORE)
#     test_dataloader_embed = make_SeqDataLoader(test_dataset_embed, batch_size=1024, shuffle=False)

    train_dataset = EmbConDataset(df=df_train, numb_samples=500000, k=1.5, internal_each_sample=INTERNAL_SAMPLE)
    validate_dataset = EmbConDataset(df=df_validate, numb_samples=300000, k=1.5)
    test_dataset = EmbConDataset(df=df_test, numb_samples=100000, k=1.5)
    
    test_dataset.sampling_validate(seed=1509, portion=1)
    test_dataloader = make_EmbConDataLoader(test_dataset, batch_size=2048)
    
    if CLASSIFY_HIDDEN is not None:
        validate_dataset_embed = EmbDataset(df_validate)
        validate_dataloader_embed = make_EmbDataLoader(validate_dataset_embed, batch_size=2048, shuffle=False)
        test_dataset_embed = EmbDataset(df_test)
        test_dataloader_embed = make_EmbDataLoader(test_dataset_embed, batch_size=2048, shuffle=False)
    
    ##### DECLARE MODEL / LOSS FUNC #####
    #con_func = TripletLoss(margin=MARGIN)
    #con_func = ContrastiveLoss_EuclidSimilarity(margin=MARGIN, max_violation=True)
    cosine_func = None #ContrastiveLoss_CosineSimilarity(margin=0.2, max_violation=False)
    con_func = SupConLoss(temperature=0.07, contrast_mode='all', base_temperature=0.07)
    cls_func = nn.BCEWithLogitsLoss().to(device)
    
    if CLASSIFY_HIDDEN is None:
        cls_func = None
        
    Model = Model_Combine(input_size=INPUT_FT, dropout=DROPOUT, emb_size=EMBEDDING_HIDDEN, 
                          projection_size=PROJECTION_OUT, cls_size=CLASSIFY_HIDDEN)
    Model = Model.to(device)

    if len(pretrained_model) > 0:
        modelCheckpoint = torch.load(pretrained_model)
        Con_Mod.load_state_dict(modelCheckpoint['model_state_dict'])
    
    params = list(filter(lambda p: p.requires_grad, Model.parameters()))
    
    if OPTIM.lower() == 'adam':
        optimizer = torch.optim.Adam(params, lr=LR)
    if OPTIM.lower() == 'sgd':
        optimizer = torch.optim.SGD(params, lr=LR)
    if USE_LARS:
        optimizer = LARS(optimizer=optimizer, eps=1e-8, trust_coef=0.001)

    ##### TRAIN #####
    min_loss = 100
    max_bacc = 0
    count_change_loss = 0
    scheduler = ReduceLROnPlateau(optimizer, factor = 0.2, patience=3, 
                                  mode = 'min', verbose=True, min_lr=1e-6)

    # Training
    for epoch in range(MAX_EPOCH):
        train_dataset.sampling_all_internal(seed=1509 + epoch, cross=True)
        train_loader = make_EmbConDataLoader(train_dataset, batch_size=4096)
        count = train_dataset.summary()
        print(Counter(count))
        validate_dataset.sampling_validate(seed=1509+epoch, portion=0.5)
        validate_dataloader = make_EmbConDataLoader(validate_dataset, batch_size=2048)
    
        # train epoch
        train_info, loss_train_dict = train_epoch_emb_combine(Model, optimizer, train_loader, con_func, loss_func_optional=cosine_func, loss_func_cls=cls_func)
        total_loss_train = loss_train_dict['total']
        euclid_loss_train = loss_train_dict['con']
        cosine_loss_train = loss_train_dict['opt']
        cls_loss_train = loss_train_dict['cls']
        
        # validate epoch
        if CLASSIFY_HIDDEN is not None:
            acc_cls_val, bacc_cls_val, f1_cls_val, dis_cls_info_val, t1, t2, t3 = classify_emb_df_combine(Model, validate_dataloader_embed)
            cls_info = f'\n[CLASSIFY VALIDATE {subject_id_validate} ONLY]\n'
            cls_info = cls_info + dis_cls_info_val + '\n'
            cls_info += f'Acc: {round(acc_cls_val, 4)}\nBAcc: {round(bacc_cls_val, 4)}\nF1: {round(f1_cls_val, 4)}'
            loss_cls_val = 2 - bacc_cls_val - f1_cls_val
        else:
            cls_info = ''
            loss_cls_val = 0
            
        val_info, loss_val_dict = validate_epoch_emb_combine(Model, validate_dataloader, con_func, loss_func_optional=cosine_func, loss_func_cls=cls_func)
        total_pro_loss_val = loss_val_dict['total']
        cls_emb_loss_val = loss_val_dict['cls']
        total_loss_val = total_pro_loss_val + cls_emb_loss_val + loss_cls_val
        
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
        str_info += val_info + '\n>>>>> SUMMARY <<<<<\n'
        
        print_info = f'Loss_Train: {round(total_loss_train, 4)}\nLoss_Val: {round(total_loss_val, 4)}'
        str_info += print_info
        str_info += cls_info

        scheduler.step(total_loss_val)
        
        print(print_info)
        
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
        
    print('Run on Test set in final')
    test_info, loss_test_dict = validate_epoch_emb_combine(Model, test_dataloader, con_func, loss_func_optional=cosine_func, loss_func_cls=cls_func)
    logging(test_info+'\n', f'{SAVE_MODEL_DIR}/{NAME_DATASET}/Log/Training_{MODEL_NAME}_{subject_id_test}.txt', True)
    
    if cls_func is not None:
        acc_cls_test, bacc_cls_test, f1_cls_test, dis_cls_info, t1, t2, t3 = classify_emb_df_combine(Model, test_dataloader_embed)
        cls_info = f'[CLASSIFY TEST {subject_id_test} ONLY]\n'
        cls_info = cls_info + dis_cls_info + '\n'
        cls_info += f'Acc: {round(acc_cls_test, 4)}\nBAcc: {round(bacc_cls_test, 4)}\nF1: {round(f1_cls_test, 4)}'
        logging(test_info+'\n', f'{SAVE_MODEL_DIR}/{NAME_DATASET}/Log/Training_{MODEL_NAME}_{subject_id_test}.txt', True)
    
def embeding_feature(train_model_path='', scaler_path=''):
    subject_id_test = scaler_path.split('/')[-1]
    subject_id_test = subject_id_test.split('_')[-1]
    subject_id_test = subject_id_test[:-7]
    print(f'Embedding Data Subject {subject_id_test}')
    ft_names = data_full.columns.tolist()

    # Scaler Data
    X = data_full.iloc[:,:-1].to_numpy()
    Y = data_full.iloc[:,-1].to_numpy()

    data_test = data_full[data_full.subject_id == subject_id_test]
    X_test = data_test.iloc[:,:-1].to_numpy()
    y_test = data_test.iloc[:,-1].to_numpy()
        
    try:
    #scaler = StandardScaler() # Need to load scaler here
        scaler = joblib.load(scaler_path)
        X[:,:-1] = scaler.transform(X[:,:-1])
        # Only for test subject
        X_test[:,:-1] = scaler.transform(X_test[:,:-1])
    except:
        print('Can find Scaler pretrained!')
        pass

    # Create Dataframe
    df_full = pd.DataFrame(data = X, columns = ft_names[:-1])
    df_full['label'] = Y
    df_test= pd.DataFrame(data = X_test, columns = ft_names[:-1])
    df_test['label'] = y_test
    
    mydataset = EmbDataset(df_full)
    mydataloader = make_EmbDataLoader(mydataset, batch_size=1024, shuffle=False)
    
    #mydataset = SequenceEmbedDataset(df_full, look_before=LOOK_BEFORE)
    #mydataloader = make_SequenceEmbedDataLoader(mydataset, batch_size=1024, shuffle=False)
    
    Model = Model_Combine(input_size=INPUT_FT, dropout=DROPOUT, emb_size=EMBEDDING_HIDDEN, 
                          projection_size=PROJECTION_OUT, cls_size=CLASSIFY_HIDDEN)
    Model = Model.to(device)
    
    modelCheckpoint = torch.load(train_model_path)
    Model.load_state_dict(modelCheckpoint['model_state_dict'])
    Model.eval()
    
    embed_ft = emb_df_combine(model=Model, dataloader=mydataloader)
    embed_ft = embed_ft.cpu().numpy()
    
    with open(f'{SAVE_MODEL_DIR}/{NAME_DATASET}/EmbedFt/EmbedFt_{MODEL_NAME}_{subject_id_test}.npy', 'wb') as f:
        np.save(f, embed_ft)    

def main():
    train_model(subject_id_test=SUBJECT_ID_TEST, 
                pretrained_model='')
    
    embeding_feature(
           train_model_path=f"{SAVE_MODEL_DIR}/{NAME_DATASET}/Model/{MODEL_NAME}_{SUBJECT_ID_TEST}.pth.tar",    
           scaler_path=f'{SAVE_MODEL_DIR}/{NAME_DATASET}/Model/StandardScaler_{SUBJECT_ID_TEST}.joblib'
    )
        
    #for idx,_ in enumerate(SUBJECT_ID_TEST):
    #    subject_id_test = SUBJECT_ID_TEST[idx]
    #    embeding_feature_and_classify(
    #        train_model_path=f"Pretrained_Cls/Model/Model_Cls_v1_{subject_id_test}.pth.tar",      
    #        scaler_path=f'Pretrained_Cls/Model/Scaler_{subject_id_test}.joblib'
    #    )
    
main()