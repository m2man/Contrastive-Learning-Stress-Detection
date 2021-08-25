import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, RobustScaler
from collections import Counter
from Model_Utils import Sequence_Model, Classify_Model, Embedding_Model, HeadProjection_Model
from Dataset_Utils import *
from General_Utils import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
import joblib
from scipy import stats

# 14, 6, 17, 3, 2, 13, 9, 10, 15, 8, 7, 11, 4, 5, 16
# SUBJECT_ID_TEST = [6, 17, 3, 2, 13, 9, 10, 15, 8, 7, 11, 4, 5]
# SUBJECT_ID_TEST = ['S'+str(x) for x in SUBJECT_ID_TEST]
SUBJECT_ID_TEST = 'S9'

# # 'GM1', 'EK1', 'NM1', 'RY1', 'KSG1', 'AD1', 'NM3', 'SJ1', 'BK1', 'RY2', 'GM2', 'MT1', 'NM2'
# SUBJECT_ID_TEST = 'AD1' # SJ1

MARGIN = 0.5 
SEQ_DIM = 128
INPUT_FT = 60
EMBEDDING_HIDDEN = [128]
PROJECTION_OUT = 60
CLASSIFY_HIDDEN = [64]
DROPOUT = 0.2
LR = 0.003
MAX_EPOCH = 50
LOOK_BEFORE = 3
INTERNAL_SAMPLE = 200

MODEL_NAME = 'EmbEuclid_sample_internal_Projection'
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

# # Take Log --> Have zeros --> Use box cox fitted
# data_full['f13'] = np.log(data_full['f13'])
# data_full['f9'] = np.log(data_full['f9'])
# data_full['f10'] = np.log(data_full['f10'])
# data_full['f3'] = np.log(data_full['f3'])
# data_full['f2'] = np.log(data_full['f2'])

def train_model(subject_id_test='S10', pretrained_con='', pretrained_pro='', pretrained_cls='', use_pro=False):
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
    
#     X_train[:,2] = np.log(X_train[:,2]+1)
#     X_train[:,3] = np.log(X_train[:,3]+1)
#     X_train[:,9] = np.log(X_train[:,9]+1)
#     X_train[:,10] = np.log(X_train[:,10]+1)
#     X_train[:,13] = np.log(X_train[:,13]+1)
    
#     X_validate[:,2] = np.log(X_validate[:,2]+1)
#     X_validate[:,3] = np.log(X_validate[:,3]+1)
#     X_validate[:,9] = np.log(X_validate[:,9]+1)
#     X_validate[:,10] = np.log(X_validate[:,10]+1)
#     X_validate[:,13] = np.log(X_validate[:,13]+1)
    
#     X_test[:,2] = np.log(X_test[:,2]+1)
#     X_test[:,3] = np.log(X_test[:,3]+1)
#     X_test[:,9] = np.log(X_test[:,9]+1)
#     X_test[:,10] = np.log(X_test[:,10]+1)
#     X_test[:,13] = np.log(X_test[:,13]+1)

    # Box-Cox
#     X_train[:,2], fitted_lambda_2 = stats.boxcox(X_train[:,2])
#     X_train[:,3], fitted_lambda_3 = stats.boxcox(X_train[:,3])
#     X_train[:,9], fitted_lambda_9 = stats.boxcox(X_train[:,9])
#     X_train[:,10], fitted_lambda_10 = stats.boxcox(X_train[:,10])
#     X_train[:,13], fitted_lambda_13 = stats.boxcox(X_train[:,13])
    
#     X_validate[:,2] = stats.boxcox(X_validate[:,2], fitted_lambda_2)
#     X_validate[:,3] = stats.boxcox(X_validate[:,3], fitted_lambda_3)
#     X_validate[:,9] = stats.boxcox(X_validate[:,9], fitted_lambda_9)
#     X_validate[:,10] = stats.boxcox(X_validate[:,10], fitted_lambda_10)
#     X_validate[:,13] = stats.boxcox(X_validate[:,13], fitted_lambda_13)
    
#     X_test[:,2] = stats.boxcox(X_test[:,2], fitted_lambda_2)
#     X_test[:,3] = stats.boxcox(X_test[:,3], fitted_lambda_3)
#     X_test[:,9] = stats.boxcox(X_test[:,9], fitted_lambda_9)
#     X_test[:,10] = stats.boxcox(X_test[:,10], fitted_lambda_10)
#     X_test[:,13] = stats.boxcox(X_test[:,13], fitted_lambda_13)
    
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
        
    validate_dataset_embed = EmbDataset(df_validate)
    validate_dataloader_embed = make_EmbDataLoader(validate_dataset_embed, batch_size=1024, shuffle=False)
    
    test_dataset_embed = EmbDataset(df_test)
    test_dataloader_embed = make_EmbDataLoader(test_dataset_embed, batch_size=1024, shuffle=False)
    
    ##### DECLARE MODEL / LOSS FUNC #####
    #con_func = TripletLoss(margin=MARGIN)
    con_func = ContrastiveLoss_EuclidSimilarity(margin=MARGIN, max_violation=False)
    #con_func = ContrastiveLoss_CosineSimilarity(margin=0.2, max_violation=True)
    #cls_func = nn.BCELoss().to(device)
    cls_func = nn.BCEWithLogitsLoss().to(device)
    
#     Con_Mod = Sequence_Model(input_dim=INPUT_FT, hidden_dim=SEQ_DIM, 
#                             numb_layers=2, dropout=DROPOUT, bidirectional=False, structure='GRU').to(device)
    Con_Mod = Embedding_Model(input_size=INPUT_FT, dropout=DROPOUT, emb_size=EMBEDDING_HIDDEN).to(device)
    Cls_Mod = Classify_Model(input_size=EMBEDDING_HIDDEN[-1], hidden_size=CLASSIFY_HIDDEN, dropout=DROPOUT).to(device)
    if use_pro:
        Pro_Mod = HeadProjection_Model(input_size=EMBEDDING_HIDDEN[-1], output_size=PROJECTION_OUT).to(device)
        if len(pretrained_pro) > 0:
            modelCheckpoint = torch.load(pretrained_pro)
            Pro_Mod.load_state_dict(modelCheckpoint['model_pro_state_dict'])
        
    if len(pretrained_con) > 0:
        modelCheckpoint = torch.load(pretrained_con)
        Con_Mod.load_state_dict(modelCheckpoint['model_con_state_dict'])
    
    if len(pretrained_cls) > 0:
        modelCheckpoint = torch.load(pretrained_cls)
        Cls_Mod.load_state_dict(modelCheckpoint['model_cls_state_dict'])
    
    con_params = list(filter(lambda p: p.requires_grad, Con_Mod.parameters()))
    cls_params = list(filter(lambda p: p.requires_grad, Cls_Mod.parameters()))
    if use_pro:
        pro_params = list(filter(lambda p: p.requires_grad, Pro_Mod.parameters()))
    else:
        pro_params = []

    optimizer = torch.optim.Adam(con_params+cls_params+pro_params, lr=LR)

    ##### TRAIN #####
    min_loss = 100
    max_bacc = 0
    count_change_loss = 0
    scheduler = ReduceLROnPlateau(optimizer, factor = 0.2, patience=2, 
                                  mode = 'min', verbose=True, min_lr=1e-6)

    # Training
    for epoch in range(MAX_EPOCH):
        train_dataset.sampling_all_internal(seed=1509 + epoch, cross=False)
        train_loader = make_EmbConDataLoader(train_dataset, batch_size=2048)
        count = train_dataset.summary()
        print(Counter(count))
        if use_pro:
            Con_Mod, Pro_Mod, Cls_Mod, optimizer, loss_total_train, loss_con_train, loss_cls_train = train_epoch_emb_cls_with_headprojection(model_con=Con_Mod, model_pro=Pro_Mod, model_cls=Cls_Mod, optimizer=optimizer, dataloader=train_loader, loss_func_cls=cls_func, loss_func_con=con_func)
        else:
            Con_Mod, Cls_Mod, optimizer, loss_total_train, loss_con_train, loss_cls_train = train_epoch_emb_cls(model_con=Con_Mod, model_cls=Cls_Mod, optimizer=optimizer, dataloader=train_loader, loss_func_cls=cls_func, loss_func_con=con_func)
        
        
        validate_dataset.shuffle(seed=1509 + epoch)
        validate_loader = make_EmbConDataLoader(validate_dataset, batch_size=2048)
    
        # Train
        # Con_Mod, Cls_Mod, optimizer, train_dataset, loss_total_train, loss_con_train, loss_cls_train = train_epoch_emb_cls_all_pairs(epoch=epoch, model_con=Con_Mod, model_cls=Cls_Mod, optimizer=optimizer, dataset=train_dataset,loss_func_cls=cls_func, loss_func_con=con_func)

        # Contrastive Val
        bacc_val, f1_val, loss_total_val, loss_con_val, loss_cls_val, dis_info = validate_epoch_emb_cls(model_con=Con_Mod, model_cls=Cls_Mod, dataloader=validate_loader, loss_func_cls=cls_func, loss_func_con=con_func)
        
        # Classify Val
        bacc_cls_val, f1_cls_val, dis_cls_info, t1, t2, t3 = classify_emb_df(Con_Mod, Cls_Mod, validate_dataloader_embed)
        cls_info = f'[CLASSIFY VAL {subject_id_validate} ONLY]\n'
        cls_info = cls_info + dis_cls_info + '\n'
        cls_info += f'BAcc: {round(bacc_cls_val, 4)}\nF1: {round(f1_cls_val, 4)}'
        
        str_info = ''

        if (bacc_cls_val+f1_cls_val)/2 > max_bacc+0.001:
            # save
            torch.save({'epoch': epoch, 
                        'model_con_state_dict': Con_Mod.state_dict(), 
                        'model_cls_state_dict': Cls_Mod.state_dict()},
                        f"{SAVE_MODEL_DIR}/{NAME_DATASET}/Model/{MODEL_NAME}_{subject_id_test}.pth.tar")
            #min_loss = loss_total_val if loss_total_val < min_loss else min_loss
            max_bacc = (bacc_cls_val+f1_cls_val)/2 
            str_info += f'[SAVE]'
            count_change_loss = 0
        else:
            if max_bacc == 1 and loss_total_val < min_loss - 0.005:
                torch.save({'epoch': epoch, 
                            'model_con_state_dict': Con_Mod.state_dict(), 
                            'model_cls_state_dict': Cls_Mod.state_dict()},
                            f"{SAVE_MODEL_DIR}/{NAME_DATASET}/Model/{MODEL_NAME}_{subject_id_test}.pth.tar")
                min_loss = loss_total_val 
                str_info += f'[SAVE]'
                count_change_loss = 0
            else:
                count_change_loss += 1
        str_info += f'===== [{epoch}/{MAX_EPOCH-1}] =====\nTotal_Loss_Train: {round(loss_total_train, 4)}\nTotal_Con_Train: {round(loss_con_train, 4)}\nTotal_Cls_Train: {round(loss_cls_train, 4)}\nTotal_Loss_Val: {round(loss_total_val, 4)}\nTotal_Con_Val: {round(loss_con_val, 4)}\nTotal_Cls_Val: {round(loss_cls_val, 4)}\nBAcc_Val: {round(bacc_val, 4)}\nF1_Val: {round(f1_val, 4)}\n{dis_info}\n{cls_info}'

        scheduler.step(loss_total_val)
        print(str_info)
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
    Con_Mod.load_state_dict(modelCheckpoint['model_con_state_dict'])
    Con_Mod.eval()
    Cls_Mod.load_state_dict(modelCheckpoint['model_cls_state_dict'])
    Cls_Mod.eval()
    
    #loss_total_test = calculate_contrastive_loss(test_dataloader, loss_func_con=cos_func)
    test_dataset.shuffle(seed=1509)
    test_loader = make_EmbConDataLoader(test_dataset, batch_size=2048)
        
    bacc_test, f1_test, loss_total_test, loss_con_test, loss_cls_test, dis_info = validate_epoch_emb_cls(model_con=Con_Mod, model_cls=Cls_Mod, dataloader=test_loader, loss_func_cls=cls_func, loss_func_con=con_func)
    
    str_loss = f'Total_Loss_Test: {round(loss_total_test, 4)}\nTotal_Con_Test: {round(loss_con_test, 4)}\nTotal_Cls_Test: {round(loss_cls_test, 4)}\nBAcc_Test: {round(bacc_test, 4)}\nF1_Test: {round(f1_test, 4)}\n{dis_info}'
    
    bacc_cls_test, f1_cls_test, dis_cls_info, t1, t2, t3 = classify_emb_df(Con_Mod, Cls_Mod, test_dataloader_embed)
    cls_info = f'[CLASSIFY TEST {subject_id_test} ONLY]\n'
    cls_info = cls_info + dis_cls_info + '\n'
    cls_info += f'BAcc: {round(bacc_cls_test, 4)}\nF1: {round(f1_cls_test, 4)}'
    
    str_loss = str_loss + '\n' + cls_info
        
    logging(str_loss, f'{SAVE_MODEL_DIR}/{NAME_DATASET}/Log/Training_{MODEL_NAME}_{subject_id_test}.txt', True)
    

def embeding_feature_and_classify(train_model_path='', scaler_path=''):
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
    
    testdataset = EmbDataset(df_test)
    testdataloader = make_EmbDataLoader(testdataset, batch_size=1024, shuffle=False)
    
    mydataset = EmbDataset(df_full)
    mydataloader = make_EmbDataLoader(mydataset, batch_size=1024, shuffle=False)
    
    #mydataset = SequenceEmbedDataset(df_full, look_before=LOOK_BEFORE)
    #mydataloader = make_SequenceEmbedDataLoader(mydataset, batch_size=1024, shuffle=False)
    
#   Con_Mod = Sequence_Model(input_dim=INPUT_FT, hidden_dim=SEQ_DIM, 
#                            numb_layers=2, dropout=DROPOUT, bidirectional=False, structure='GRU')
    Con_Mod = Embedding_Model(input_size=INPUT_FT, dropout=DROPOUT, emb_size=EMBEDDING_HIDDEN).to(device)
    Cls_Mod = Classify_Model(input_size=EMBEDDING_HIDDEN[-1], hidden_size=CLASSIFY_HIDDEN, dropout=DROPOUT).to(device)
    
    modelCheckpoint = torch.load(train_model_path)
    Con_Mod.load_state_dict(modelCheckpoint['model_con_state_dict'])
    Cls_Mod.load_state_dict(modelCheckpoint['model_cls_state_dict'])
    Con_Mod.eval()
    Cls_Mod.eval()
    
    bacc, f1, dis_info, embed_ft, preds, labels = classify_emb_df(Con_Mod, Cls_Mod, mydataloader)
    embed_ft = embed_ft.cpu().numpy()
    
    info = f'===== [CLASSIFY RESULT] =====\n'
    info = info + dis_info + '\n'
    info += f'All BAcc: {round(bacc, 4)}\nAll F1: {round(f1, 4)}\n'
    
    bacc, f1, dis_info, t1, t2, t3 = classify_emb_df(Con_Mod, Cls_Mod, testdataloader)
    info += f'===== [CLASSIFY RESULT {subject_id_test} ONLY] =====\n'
    info = info + dis_info + '\n'
    info += f'BAcc: {round(bacc, 4)}\nF1: {round(f1, 4)}\n'
    
    logging(info, f'{SAVE_MODEL_DIR}/{NAME_DATASET}/Log/Testing_{MODEL_NAME}_{subject_id_test}.txt', True)
    
    with open(f'{SAVE_MODEL_DIR}/{NAME_DATASET}/EmbedFt/EmbedFt_{MODEL_NAME}_{subject_id_test}.npy', 'wb') as f:
        np.save(f, embed_ft)

def main():
    train_model(subject_id_test=SUBJECT_ID_TEST, 
                pretrained_con='',
                pretrained_cls='',
                pretrained_pro='',
                use_pro=True)
    
#     embeding_feature_and_classify(
#            train_model_path=f"{SAVE_MODEL_DIR}/{NAME_DATASET}/Model/{MODEL_NAME}_{SUBJECT_ID_TEST}.pth.tar",    
#            scaler_path=f'{SAVE_MODEL_DIR}/{NAME_DATASET}/Model/StandardScaler_{SUBJECT_ID_TEST}.joblib'
#     )
        
    #for idx,_ in enumerate(SUBJECT_ID_TEST):
    #    subject_id_test = SUBJECT_ID_TEST[idx]
    #    embeding_feature_and_classify(
    #        train_model_path=f"Pretrained_Cls/Model/Model_Cls_v1_{subject_id_test}.pth.tar",      
    #        scaler_path=f'Pretrained_Cls/Model/Scaler_{subject_id_test}.joblib'
    #    )
    
main()