import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit
from tqdm import tqdm
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from collections import Counter
from Contrastive_Utils import *
import joblib

# 14, 6, 17, 3, 2, 13, 9, 10, 15, 8, 7, 11, 4, 5, 16
SUBJECT_ID_TEST = 'S16'

#### V1 ####
MARGIN = 0.1
CONTRASTIVE_HIDDEN = [128]
INPUT_FT = 60
DROPOUT = 0.15
LR = 0.003
MAX_EPOCH = 40

##### READ DATASET #####
DATA_DIR = '/home/nvtu/PhD_Work/StressDetection/DATA/MyDataset/WESAD'
NAME_DATASET = DATA_DIR.split('/')[-1]

data_group = np.load(f'{DATA_DIR}/{NAME_DATASET}_WRIST_groups_1.npy')
data_gt = np.load(f'{DATA_DIR}/{NAME_DATASET}_WRIST_ground_truth_1.npy')
data_ft = np.load(f'{DATA_DIR}/{NAME_DATASET}_WRIST_stats_feats_1.npy')
# with open('test.npy', 'wb') as f:
#     np.save(f, np.array([1, 2]))

# Create dataframe for dataset
column_values = [f'f{x}' for x in range(data_ft.shape[1])]
data_full = pd.DataFrame(data = data_ft,  
                         columns = column_values)
data_full['subject_id'] = data_group
data_full['label'] = data_gt
list_subject_id = np.unique(data_full['subject_id']).tolist()

def train_model(subject_id_test='S10'):
    ##### TRAIN / VAL / TEST #####
    # Currently general without using any person specific
    # subject_id_test = 'S10'
    data_train = data_full[data_full.subject_id != subject_id_test]
    data_test = data_full[data_full.subject_id == subject_id_test]
    ft_names = data_train.columns.tolist()

    # Scaler Data
    X_train_val = data_train.iloc[:,:-1].to_numpy()
    y_train_val = data_train.iloc[:,-1].to_numpy()
    X_test = data_test.iloc[:,:-1].to_numpy()
    y_test = data_test.iloc[:,-1].to_numpy()

    validate_portion = 0.15
    X_train, X_validate, y_train, y_validate = train_test_split(X_train_val, y_train_val, 
                                                                test_size=validate_portion, 
                                                                random_state=1509, stratify=y_train_val)

    scaler = StandardScaler()
    X_train[:,:-1] = scaler.fit_transform(X_train[:,:-1])
    X_validate[:,:-1] = scaler.transform(X_validate[:,:-1])
    X_test[:,:-1] = scaler.transform(X_test[:,:-1])
    joblib.dump(scaler, f'Pretrained/Model/Scaler_{subject_id_test}.joblib')

    # Create Dataframe
    df_train = pd.DataFrame(data = X_train, columns = ft_names[:-1])
    df_train['label'] = y_train

    df_validate = pd.DataFrame(data = X_validate, columns = ft_names[:-1])
    df_validate['label'] = y_validate

    df_test= pd.DataFrame(data = X_test, columns = ft_names[:-1])
    df_test['label'] = y_test

    ##### DATASET / DATALOADER #####
    train_dataset = ContrastiveDataset(df=df_train, numb_samples=600000, k=1.5)
    validate_dataset = ContrastiveDataset(df=df_validate, numb_samples=100000, k=1.5)
    

    ##### DECLARE MODEL / LOSS FUNC #####
    cos_func = ContrastiveLoss_CosineSimilarity(margin=MARGIN, max_violation=True)

    Con_Mod = Contrastive_Model(input_size=INPUT_FT, emb_size=CONTRASTIVE_HIDDEN, dropout=DROPOUT).to(device)

    con_params = list(filter(lambda p: p.requires_grad, Con_Mod.parameters()))

    optimizer = torch.optim.Adam(con_params, lr=LR)

    ##### TRAIN #####
    min_loss = 100
    count_change_loss = 0
    scheduler = ReduceLROnPlateau(optimizer, factor = 0.2, patience=2, 
                                  mode = 'min', verbose=True, min_lr=1e-6)

    # Training
    for epoch in range(MAX_EPOCH):
        train_dataset.shuffle(seed=1509 + epoch)
        train_loader = make_ContrastiveDataLoader(train_dataset, batch_size=2048)
        
        validate_dataset.shuffle(seed=1509 + epoch)
        validate_dataloader = make_ContrastiveDataLoader(validate_dataset, batch_size=2048)
        
        Con_Mod, optimizer, loss_total_train = train_epoch_contrastive(model_con=Con_Mod, optimizer=optimizer, dataloader=train_loader,loss_func_con=cos_func)

        loss_total_val = validate_contrastive(model_con=Con_Mod, dataloader=validate_dataloader, loss_func_con=cos_func)

        str_info = ''

        if loss_total_val < min_loss:
            # save
            torch.save({'epoch': epoch, 'model_con_state_dict': Con_Mod.state_dict()}, 
                       f"Pretrained/Model/Model_Con_v1_{subject_id_test}.pth.tar")
            min_loss = loss_total_val if loss_total_val < min_loss else min_loss
            str_info += f'[SAVE]'
            count_change_loss = 0
        else:
            count_change_loss += 1
        str_info += f'===== [{epoch}/{MAX_EPOCH-1}] =====\nTotal_Loss_Train: {round(loss_total_train, 4)}\nTotal_Loss_Val: {round(loss_total_val, 4)}\n'

        scheduler.step(loss_total_val)

        logging(str_info, f'Pretrained/Log/Training_Con_v1_{subject_id_test}.txt', True)

        if count_change_loss >= 7:
            logging('Early Stopping', f'Pretrained/Log/Training_Con_v1_{subject_id_test}.txt', True)
            break

    #Test
    modelCheckpoint = torch.load(f"Pretrained/Model/Model_Con_v1_{subject_id_test}.pth.tar")
    Con_Mod.load_state_dict(modelCheckpoint['model_con_state_dict'])
    Con_Mod.eval()
    test_dataset = ContrastiveDataset(df=df_test, numb_samples=100000, k=1.5)
    test_dataset.shuffle(seed=1509)
    test_dataloader = make_ContrastiveDataLoader(test_dataset, batch_size=2048)
    #loss_total_test = calculate_contrastive_loss(test_dataloader, loss_func_con=cos_func)
    loss_total_test = validate_contrastive(model_con=Con_Mod, dataloader=test_dataloader, loss_func_con=cos_func)
    
    str_loss = f'Test Loss: {round(loss_total_test, 4)}'
    logging(str_loss, f'Pretrained/Log/Training_Con_v1_{subject_id_test}.txt', True)
    
def embeding_feature(train_model_path='', scaler_path=''):
    ft_names = data_full.columns.tolist()

    # Scaler Data
    X = data_full.iloc[:,:-1].to_numpy()
    Y = data_full.iloc[:,-1].to_numpy()

    #scaler = StandardScaler() # Need to load scaler here
    scaler = joblib.load(scaler_path)
    X[:,:-1] = scaler.transform(X[:,:-1])

    # Create Dataframe
    df_full = pd.DataFrame(data = X, columns = ft_names[:-1])
    df_full['label'] = Y
    
    mydataset = EmbedDataset(df_full)
    mydataloader = make_EmbedDataLoader(mydataset, batch_size=1024, shuffle=False)
    
    Con_Mod = Contrastive_Model(input_size=INPUT_FT, emb_size=CONTRASTIVE_HIDDEN, dropout=DROPOUT).to(device)
    modelCheckpoint = torch.load(train_model_path)
    Con_Mod.load_state_dict(modelCheckpoint['model_con_state_dict'])
    Con_Mod.eval()
    
    embed_ft, labels = embed_df(Con_Mod, mydataloader)
    embed_ft = embed_ft.cpu().numpy()
    
    with open(f'EmbededFt/{NAME_DATASET}_contrastive_embed.npy', 'wb') as f:
        np.save(f, embed_ft)

def main():
    subject_id_test = SUBJECT_ID_TEST
    train_model(subject_id_test=subject_id_test)
    #embeding_feature(train_model_path=f"Output_Contrastive/Model/Model_Con_v2_{subject_id_test}.pth.tar",
    #                scaler_path=f'Output_Contrastive/Model/Scaler_{subject_id_test}.joblib')
    
main()