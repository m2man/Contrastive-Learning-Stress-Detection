import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda:0')
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from torch.autograd import Variable
import joblib
sigmoid_function = nn.Sigmoid()
torch.autograd.set_detect_anomaly(True)
from Dataset_Utils import make_EmbConDataLoader
import math

def logging(s, path, print_=False):
    if print_:
        print(s)
    if path:
        with open(path, 'a+') as f:
            f.write(s + '\n')

# def adjust_learning_rate(args, optimizer, epoch):
#     lr = args.learning_rate
#     if args.cosine:
#         eta_min = lr * (args.lr_decay_rate ** 3)
#         lr = eta_min + (lr - eta_min) * (
#                 1 + math.cos(math.pi * epoch / args.epochs)) / 2
#     else:
#         steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
#         if steps > 0:
#             lr = lr * (args.lr_decay_rate ** steps)

#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


# def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
#     if args.warm and epoch <= args.warm_epochs:
#         p = (batch_id + (epoch - 1) * total_batches) / \
#             (args.warm_epochs * total_batches)
#         lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

#         for param_group in optimizer.param_groups:
#             param_group['lr'] = lr

            
####################################################################################    
def train_epoch_seq_cls_triplet(model_con, model_cls, optimizer, dataloader, loss_func_con, loss_func_cls):
    '''
    Train 1 epoch
    '''
    total_loss = []
    con_loss = []
    cls_loss = []
    model_con.train()
    model_cls.train()
    for i, (samples_1, samples_2, samples_3, len_1, len_2, len_3, label_1, label_2, label_3) in enumerate(dataloader):
        samples_1 = samples_1.to(device)
        samples_2 = samples_2.to(device)
        samples_3 = samples_3.to(device)
        #label_1 = label_1.to(device)
        labels_cat = torch.cat((label_1, label_3)).unsqueeze(-1).to(device)
        
        optimizer.zero_grad()
        # Contrastive Model (Embedding)
        samples_1_emb = model_con(samples_1, len_1)
        samples_2_emb = model_con(samples_2, len_2)
        samples_3_emb = model_con(samples_3, len_3)
        
        # Classify Model
        samples_cat_emb = torch.cat((samples_1_emb, samples_3_emb))
        preds_cls = model_cls(samples_cat_emb)
        
        loss_con = loss_func_con(samples_1_emb, samples_2_emb, samples_3_emb)
        loss_cls = loss_func_cls(preds_cls, labels_cat)
        loss_total = loss_con + loss_cls
        
        # Update Grad
        loss_total.backward()
        optimizer.step()
        
        total_loss.append(loss_total.item())
        con_loss.append(loss_con.item())
        cls_loss.append(loss_cls.item())
        
        str_info = f'[Iter {i}/{len(dataloader)}] Total: {round(np.mean(total_loss), 4)} -- Con: {round(np.mean(con_loss), 4)} -- Cls: {round(np.mean(cls_loss), 4)}'
        if i % 250 == 0:
            print(str_info)
            # logging(str_info, 'log.txt')
    print('Done Epoch')
    return model_con, model_cls, optimizer, np.mean(total_loss), np.mean(con_loss), np.mean(cls_loss)

def validate_epoch_seq_cls_triplet(model_con, model_cls, dataloader, loss_func_con, loss_func_cls):
    '''
    Validate model
    '''
    total_loss = []
    con_loss = []
    cls_loss = []
    total_pred = []
    total_label = []
    with torch.no_grad():
        model_con.eval()
        model_cls.eval()
        for i, (samples_1, samples_2, samples_3, len_1, len_2, len_3, label_1, label_2, label_3) in enumerate(dataloader):
            samples_1 = samples_1.to(device)
            samples_2 = samples_2.to(device)
            samples_3 = samples_3.to(device)
            #label_1 = label_1.to(device)
            labels_cat = torch.cat((label_1, label_3)).unsqueeze(-1).to(device)

            # Contrastive Model (Embedding)
            samples_1_emb = model_con(samples_1, len_1)
            samples_2_emb = model_con(samples_2, len_2)
            samples_3_emb = model_con(samples_3, len_3)
            
            # Classify Model
            samples_cat_emb = torch.cat((samples_1_emb, samples_3_emb))
            preds_cls = model_cls(samples_cat_emb)

            loss_con = loss_func_con(samples_1_emb, samples_2_emb, samples_3_emb)
            loss_cls = loss_func_cls(preds_cls, labels_cat)
            loss_total = loss_con + loss_cls
            
            total_loss.append(loss_total.item())
            con_loss.append(loss_con.item())
            cls_loss.append(loss_cls.item())
        
            preds = preds_cls.round().squeeze().detach().cpu().numpy().tolist()
            labels = labels_cat.round().squeeze().detach().cpu().numpy().tolist()
            total_pred += preds
            total_label += labels
                        
        total_loss = np.mean(total_loss)
        con_loss = np.mean(con_loss)
        cls_loss = np.mean(cls_loss)
        dis_info = f"Class Samples Val: {len(total_label) - np.sum(total_label)} / {np.sum(total_label)}"
        acc = balanced_accuracy_score(total_label, total_pred)
        f1 = f1_score(total_label, total_pred)
    return acc, f1, total_loss, con_loss, cls_loss, dis_info

def classify_seq_df(model_con, model_cls, dataloader):
    all_ft_embed = []
    all_pred = []
    all_label = []
    with torch.no_grad():
        model_con.eval()
        model_cls.eval()
        
        for i, (samples, len_samples, labels) in enumerate(dataloader):
            samples = samples.to(device)
            labels = labels.to(device)
            labels = labels.unsqueeze(-1)
            
            samples_emb = model_con(samples, len_samples)
            preds_cls = model_cls(samples_emb)
            
            preds = preds_cls.round().squeeze().detach().cpu().numpy().tolist()
            labels = labels.round().squeeze().detach().cpu().numpy().tolist()
            all_pred += preds
            all_label += labels
            
            samples_emb = samples_emb.cpu()
            all_ft_embed.append(samples_emb)
            
        dis_info = f"Class Samples Distribution: {len(all_label) - np.sum(all_label)} / {np.sum(all_label)}"
        acc = balanced_accuracy_score(all_label, all_pred)
        f1 = f1_score(all_label, all_pred)    
        all_ft_embed = torch.cat(all_ft_embed)
    return acc, f1, dis_info, all_ft_embed, all_pred, all_label


####################################################################################
def train_epoch_emb_cls(model_con, model_cls, optimizer, dataloader, loss_func_con, loss_func_cls):
    '''
    Train 1 epoch
    '''
    total_loss = []
    con_loss = []
    cls_loss = []
    model_con.train()
    model_cls.train()
    for i, (samples_1, samples_2, labels) in enumerate(dataloader):
        samples_1 = samples_1.to(device)
        samples_2 = samples_2.to(device)
        labels = labels.to(device)
        labels_cat = torch.cat((labels, labels)).unsqueeze(-1)
        
        optimizer.zero_grad()
        # Contrastive Model (Embedding)
        samples_1_emb = model_con(samples_1)
        samples_2_emb = model_con(samples_2)
        
        # Classify Model
        samples_cat_emb = torch.cat((samples_1_emb, samples_2_emb))
        preds_cls = model_cls(samples_cat_emb)
        
        loss_con = loss_func_con(samples_1_emb, samples_2_emb)
        loss_cls = loss_func_cls(preds_cls, labels_cat)
        loss_total = loss_con + 2*loss_cls
        
        # Update Grad
        loss_total.backward()
        optimizer.step()
            
        total_loss.append(loss_total.item())
        con_loss.append(loss_con.item())
        cls_loss.append(loss_cls.item())
 
        str_info = f'[Iter {i}/{len(dataloader)}] Total: {round(np.mean(total_loss), 4)} -- Con: {round(np.mean(con_loss), 4)} -- Cls: {round(np.mean(cls_loss), 4)}'
        if i % 250 == 0 or i == len(dataloader)-1:
            print(str_info)
            # logging(str_info, 'log.txt')
    print('Done Epoch')
    return model_con, model_cls, optimizer, np.mean(total_loss), np.mean(con_loss), np.mean(cls_loss)

def train_epoch_emb_cls_with_headprojection(model_con, model_pro, model_cls, optimizer, dataloader, loss_func_con, loss_func_cls):
    '''
    Train 1 epoch
    '''
    total_loss = []
    con_loss = []
    cls_loss = []
    model_con.train()
    model_cls.train()
    model_pro.train()
    for i, (samples_1, samples_2, labels) in enumerate(dataloader):
        samples_1 = samples_1.to(device)
        samples_2 = samples_2.to(device)
        labels = labels.to(device)
        labels_cat = torch.cat((labels, labels)).unsqueeze(-1)
        
        optimizer.zero_grad()
        # Contrastive Model (Embedding)
        samples_1_emb = model_con(samples_1)
        samples_2_emb = model_con(samples_2)
        
        # Classify Model
        samples_cat_emb = torch.cat((samples_1_emb, samples_2_emb))
        preds_cls = model_cls(samples_cat_emb)
        
        # Head Projection
        samples_1_pro = model_pro(samples_1_emb)
        samples_2_pro = model_pro(samples_2_emb)
        
        loss_con = loss_func_con(samples_1_pro, samples_2_pro)
        loss_cls = loss_func_cls(preds_cls, labels_cat)
        loss_total = loss_con + loss_cls
        
        # Update Grad
        loss_total.backward()
        optimizer.step()
            
        total_loss.append(loss_total.item())
        con_loss.append(loss_con.item())
        cls_loss.append(loss_cls.item())
 
        str_info = f'[Iter {i}/{len(dataloader)}] Total: {round(np.mean(total_loss), 4)} -- Con: {round(np.mean(con_loss), 4)} -- Cls: {round(np.mean(cls_loss), 4)}'
        if i % 250 == 0 or i == len(dataloader)-1:
            print(str_info)
            # logging(str_info, 'log.txt')
    print('Done Epoch')
    return model_con, model_pro, model_cls, optimizer, np.mean(total_loss), np.mean(con_loss), np.mean(cls_loss)

def train_epoch_emb_cls_all_pairs(epoch, model_con, model_cls, optimizer, dataset, loss_func_con, loss_func_cls):
    '''
    Train 1 epoch
    '''
    total_loss = []
    con_loss = []
    cls_loss = []
    previous_loss = 100
    model_con.train()
    model_cls.train()
    finish = False
    count_iter = 0
    while not finish:
        dataset.sampling_all(seed=1509 + epoch)
        if dataset.all_is_process:
            finish = True
        dataloader = make_EmbConDataLoader(dataset, batch_size=2048) 
        count_iter += len(dataloader)
        for i, (samples_1, samples_2, labels) in enumerate(dataloader):
            samples_1 = samples_1.to(device)
            samples_2 = samples_2.to(device)
            labels = labels.to(device)
            labels_cat = torch.cat((labels, labels)).unsqueeze(-1)

            optimizer.zero_grad()
            # Contrastive Model (Embedding)
            samples_1_emb = model_con(samples_1)
            samples_2_emb = model_con(samples_2)

            # Classify Model
            samples_cat_emb = torch.cat((samples_1_emb, samples_2_emb))
            preds_cls = model_cls(samples_cat_emb)

            loss_con = loss_func_con(samples_1_emb, samples_2_emb)
            loss_cls = loss_func_cls(preds_cls, labels_cat)
            loss_total = loss_con + 2*loss_cls

            # Update Grad
            loss_total.backward()
            optimizer.step()
            
            total_loss.append(loss_total.item())
            con_loss.append(loss_con.item())
            cls_loss.append(loss_cls.item())
 
            str_info = f'[Iter {i}/{len(dataloader)}] Total: {round(np.mean(total_loss), 4)} -- Con: {round(np.mean(con_loss), 4)} -- Cls: {round(np.mean(cls_loss), 4)}'
            if i % 250 == 0 or i == len(dataloader)-1:
                print(str_info)
                # logging(str_info, 'log.txt')
        
        # Converge in this epoch
        current_loss = np.mean(total_loss)
        if previous_loss - current_loss < 0.001: 
            finish = True
            dataset.all_is_process = True
        else:
            previous_loss = current_loss
            
    print(f'Done Epoch with {count_iter} iterations')
    return model_con, model_cls, optimizer, dataset, np.mean(total_loss), np.mean(con_loss), np.mean(cls_loss)

def validate_epoch_emb_cls(model_con, model_cls, dataloader, loss_func_con, loss_func_cls):
    '''
    Validate model
    '''
    total_loss = []
    con_loss = []
    cls_loss = []
    total_pred = []
    total_label = []
    with torch.no_grad():
        model_con.eval()
        model_cls.eval()
        for i, (samples_1, samples_2, labels) in enumerate(dataloader):
            samples_1 = samples_1.to(device)
            samples_2 = samples_2.to(device)
            labels = labels.to(device)
            labels_cat = torch.cat((labels, labels)).unsqueeze(-1).to(device)

            # Contrastive Model (Embedding)
            samples_1_emb = model_con(samples_1)
            samples_2_emb = model_con(samples_2)
            
            # Classify Model
            samples_cat_emb = torch.cat((samples_1_emb, samples_2_emb))
            preds_cls = model_cls(samples_cat_emb)
            
            loss_con = loss_func_con(samples_1_emb, samples_2_emb)
            loss_cls = loss_func_cls(preds_cls, labels_cat)
            loss_total = loss_con + 2*loss_cls
            
            total_loss.append(loss_total.item())
            con_loss.append(loss_con.item())
            cls_loss.append(loss_cls.item())
            
            preds_cls = sigmoid_function(preds_cls)
            preds = preds_cls.round().squeeze().detach().cpu().numpy().tolist()
            labels = labels_cat.round().squeeze().detach().cpu().numpy().tolist()
            total_pred += preds
            total_label += labels
                        
        total_loss = np.mean(total_loss)
        con_loss = np.mean(con_loss)
        cls_loss = np.mean(cls_loss)
        dis_info = f"Class Samples Val: {len(total_label) - np.sum(total_label)} / {np.sum(total_label)}"
        acc = balanced_accuracy_score(total_label, total_pred)
        f1 = f1_score(total_label, total_pred)
    return acc, f1, total_loss, con_loss, cls_loss, dis_info

def classify_emb_df(model_con, model_cls, dataloader):
    all_ft_embed = []
    all_pred = []
    all_label = []
    with torch.no_grad():
        model_con.eval()
        model_cls.eval()
        
        for i, (samples, labels) in enumerate(dataloader):
            samples = samples.to(device)
            labels = labels.to(device)
            labels = labels.unsqueeze(-1)
            
            samples_emb = model_con(samples)
            preds_cls = model_cls(samples_emb)
            
            preds_cls = sigmoid_function(preds_cls)
            preds = preds_cls.round().squeeze().detach().cpu().numpy().tolist()
            labels = labels.round().squeeze().detach().cpu().numpy().tolist()
            all_pred += preds
            all_label += labels
            
            samples_emb = samples_emb.cpu()
            all_ft_embed.append(samples_emb)
            
        dis_info = f"Class Samples Distribution: {len(all_label) - np.sum(all_label)} / {np.sum(all_label)}"
        acc = balanced_accuracy_score(all_label, all_pred)
        f1 = f1_score(all_label, all_pred)    
        all_ft_embed = torch.cat(all_ft_embed)
    return acc, f1, dis_info, all_ft_embed, all_pred, all_label



################################## ONLY EMBEDDING ##################################
def train_epoch_emb_with_headprojection(model_con, model_pro, optimizer, dataloader, loss_func_con):
    '''
    Train 1 epoch
    '''
    total_loss = []
    model_con.train()
    model_pro.train()
    for i, (samples_1, samples_2, labels, idx_0, idx_1) in enumerate(dataloader):
        samples_1 = samples_1.to(device)
        samples_2 = samples_2.to(device)
        
        optimizer.zero_grad()
        # Contrastive Model (Embedding)
        samples_1_emb = model_con(samples_1)
        samples_2_emb = model_con(samples_2)
        
        # Head Projection
        samples_1_pro = model_pro(samples_1_emb)
        samples_2_pro = model_pro(samples_2_emb)
        
        loss_total = loss_func_con(samples_1_pro, samples_2_pro)
        #print(samples_1_pro)
        #print(samples_2_pro)
        #print(labels)
        
        # Update Grad
        loss_total.backward()
        optimizer.step()
            
        total_loss.append(loss_total.item())
 
        str_info = f'[Iter {i}/{len(dataloader)}] Total: {round(np.mean(total_loss), 4)}'
        if i % 250 == 0 or i == len(dataloader)-1:
            print(str_info)
            # break # for debug
    #print(idx_0)
    #print(idx_1)
    print('===== Samples 1 =====')
    print(samples_1)
    print('===== Samples 1 Emb =====')
    print(samples_1_emb)
    print('===== Samples 1 Pro =====')
    print(samples_1_pro)
    print('===== Samples 2 =====')
    print(samples_2)
    print('===== Samples 2 Emb =====')
    print(samples_2_emb)
    print('===== Samples 2 Pro =====')
    print(samples_2_pro)
    print('Done Epoch')
    return model_con, model_pro, optimizer, np.mean(total_loss)

def train_epoch_emb(model_con, optimizer, dataloader, loss_func_con):
    '''
    Train 1 epoch
    '''
    total_loss = []
    model_con.train()
    for i, (samples_1, samples_2, labels) in enumerate(dataloader):
        samples_1 = samples_1.to(device)
        samples_2 = samples_2.to(device)
        
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
        if i % 250 == 0 or i == len(dataloader)-1:
            print(str_info)
    
    print('Done Epoch')
    return model_con, optimizer, np.mean(total_loss)

def validate_epoch_emb_with_headprojection(model_con, model_pro, dataloader, loss_func_con):
    '''
    Validate model
    '''
    con_loss = []
    pro_loss = []
    total_label = []
    with torch.no_grad():
        model_con.eval()
        model_pro.eval()
        for i, (samples_1, samples_2, labels, idx_0, idx_1) in enumerate(dataloader):
            samples_1 = samples_1.to(device)
            samples_2 = samples_2.to(device)
            labels = labels.to(device)
            
            # Contrastive Model (Embedding)
            samples_1_emb = model_con(samples_1)
            samples_2_emb = model_con(samples_2)

            # Head Projection
            samples_1_pro = model_pro(samples_1_emb)
            samples_2_pro = model_pro(samples_2_emb)
            
            loss_con = loss_func_con(samples_1_emb, samples_2_emb)
            loss_pro = loss_func_con(samples_1_pro, samples_2_pro)
            
            con_loss.append(loss_con.item())
            pro_loss.append(loss_pro.item())
            
            labels = labels.round().detach().cpu().numpy().tolist()
            total_label += labels
            
            str_info = f'[Iter {i}/{len(dataloader)}] Loss_Con: {round(np.mean(con_loss), 4)} -- Loss_Pro: {round(np.mean(pro_loss), 4)}'
            if i % 200 == 0 or i == len(dataloader)-1:
                print(str_info)
            
        con_loss = np.mean(con_loss)
        pro_loss = np.mean(pro_loss)
        dis_info = f"Pairs Dist: {len(total_label) - np.sum(total_label)} / {np.sum(total_label)}"
        
    return con_loss, pro_loss, dis_info

def validate_epoch_emb(model_con, dataloader, loss_func_con):
    '''
    Validate model
    '''
    con_loss = []
    total_label = []
    with torch.no_grad():
        model_con.eval()
        model_pro.eval()
        for i, (samples_1, samples_2, labels) in enumerate(dataloader):
            samples_1 = samples_1.to(device)
            samples_2 = samples_2.to(device)
            labels = labels.to(device)
            
            # Contrastive Model (Embedding)
            samples_1_emb = model_con(samples_1)
            samples_2_emb = model_con(samples_2)
            
            loss_con = loss_func_con(samples_1_emb, samples_2_emb)
            
            con_loss.append(loss_con.item())
            
            labels = labels.round().detach().cpu().numpy().tolist()
            total_label += labels

        con_loss = np.mean(con_loss)
        dis_info = f"Pairs Dist: {len(total_label) - np.sum(total_label)} / {np.sum(total_label)}"
        
    return con_loss, dis_info


def emb_df(model_con, model_pro=None, dataloader=None):
    all_ft_embed = []
    with torch.no_grad():
        model_con.eval()
        if model_pro is not None:
            model_pro.eval()
            
        for i, (samples, labels) in enumerate(dataloader):
            samples = samples.to(device)
            samples_emb = model_con(samples)        
            if model_pro is not None:
                samples_emb = model_pro(samples_emb)
            samples_emb = samples_emb.cpu()
            all_ft_embed.append(samples_emb)
                
        all_ft_embed = torch.cat(all_ft_embed)
    return all_ft_embed
####################################################################






################################## COMBINE EMBEDING ##################################
def train_epoch_emb_combine(model, optimizer, dataloader, loss_func_con, loss_func_cls=None, loss_func_optional=None):
    '''
    Train 1 epoch
    '''
    total_loss = []
    con_loss = []
    option_loss = []
    cls_loss = []
    dis_info = ''
    model.train()
    for i, (samples_1, samples_2, labels, idx_0, idx_1) in enumerate(dataloader):
        samples_1 = samples_1.to(device)
        samples_2 = samples_2.to(device)
        labels = labels.to(device)
        labels_cat = torch.cat((labels, labels)).unsqueeze(-1)
        
        bsz = len(labels)
        
        samples = torch.cat((samples_1, samples_2))
        
        samples_pro, samples_emb, samples_cls = model(samples)
        samples_1_pro, samples_2_pro = torch.split(samples_pro, [bsz,bsz], dim=0)
        samples_1_emb, samples_2_emb = torch.split(samples_emb, [bsz,bsz], dim=0)
        
        # SupConLoss
        #features = torch.cat([samples_1_pro.unsqueeze(1), samples_2_pro.unsqueeze(1)], dim=1)
        #loss_con = loss_func_con(features, labels)
        
        # EuclidLoss
        loss_con = loss_func_con(samples_1_pro, samples_2_pro)
        
        if loss_func_optional is not None:
            loss_option = loss_func_optional(samples_1_pro, samples_2_pro)
        else:
            loss_option = 0
        if loss_func_cls is not None:
            loss_cls = loss_func_cls(samples_cls, labels_cat)
        else:
            loss_cls = 0
        
        loss_total = loss_con + loss_option + loss_cls
            
        # Update Grad
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
            
        total_loss.append(loss_total.item())
        con_loss.append(loss_con.item())
        str_info = f'[Iter {i}/{len(dataloader)}] Total: {round(np.mean(total_loss), 4)}'
        
        if loss_func_optional is not None or loss_func_cls is not None:
            str_info += f' -- Con: {round(np.mean(con_loss), 4)}'
            if loss_func_cls is not None:
                cls_loss.append(loss_cls.item())
                str_info += f' -- Cls: {round(np.mean(cls_loss), 4)}'
            else:
                cls_loss.append(loss_cls)
                
            if loss_func_optional is not None:
                option_loss.append(loss_option.item())
                str_info += f' -- Opt: {round(np.mean(option_loss), 4)}'
            else:
                option_loss.append(loss_option)
        else:
            cls_loss.append(loss_cls)
            option_loss.append(loss_option)
            
        if i % 250 == 0 or i == len(dataloader)-1:
            print(str_info)
            # break # for debug
            
            dis_info += str_info + '\n'
    
    loss_dict = {}
    loss_dict['total'] = np.mean(total_loss)
    loss_dict['con'] = np.mean(con_loss) 
    loss_dict['opt'] = np.mean(option_loss) 
    loss_dict['cls'] = np.mean(cls_loss) 
    
    print('Done Epoch')
    return dis_info, loss_dict

def validate_epoch_emb_combine(model, dataloader, loss_func_con, loss_func_optional=None, loss_func_cls=None):
    '''
    Validate model
    '''
    con_pro_loss = []
    cls_emb_loss = []
    opt_pro_loss = []
    total_pro_loss = []
    total_label = []
    dis_info = ''
    with torch.no_grad():
        model.eval()
        for i, (samples_1, samples_2, labels, idx_0, idx_1) in enumerate(dataloader):
            samples_1 = samples_1.to(device)
            samples_2 = samples_2.to(device)
            labels = labels.to(device)
            labels_cat = torch.cat((labels, labels)).unsqueeze(-1)
            bsz = len(labels)
            
            samples = torch.cat((samples_1, samples_2))
            samples_pro, samples_emb, samples_cls = model(samples)
            samples_1_pro, samples_2_pro = torch.split(samples_pro, [bsz,bsz], dim=0)
            #samples_1_emb, samples_2_emb = torch.split(samples_emb, [bsz,bsz], dim=0)

            # SupConLoss
            #features_pro = torch.cat([samples_1_pro.unsqueeze(1), samples_2_pro.unsqueeze(1)], dim=1)
            ##features_emb = torch.cat([samples_1_emb.unsqueeze(1), samples_2_emb.unsqueeze(1)], dim=1)
            
            # loss_emb_con = loss_func_con(features_emb, labels)
            #loss_pro_con = loss_func_con(features_pro, labels)
            
            # EuclidLoss
            loss_pro_con = loss_func_con(samples_1_pro, samples_2_pro)
        
            if loss_func_cls is not None:
                loss_emb_cls = loss_func_cls(samples_cls, labels_cat)
                cls_emb_loss.append(loss_emb_cls.item())
            else:
                loss_emb_cls = 0
                cls_emb_loss.append(loss_emb_cls)
                
            if loss_func_optional is not None:
                #loss_con_option = loss_func_optional(samples_1_emb, samples_2_emb)
                loss_pro_option = loss_func_optional(samples_1_pro, samples_2_pro)
                opt_pro_loss.append(loss_pro_option.item())
            else:
                #loss_con_option = 0
                loss_pro_option = 0
                opt_pro_loss.append(loss_pro_option)
            
            #loss_con_total = loss_con + loss_con_option
            loss_pro_total = loss_pro_con + loss_pro_option
            
            #con_emb_loss.append(loss_con.item())
            con_pro_loss.append(loss_pro_con.item())
            
            #total_con_loss.append(loss_con_total.item())
            total_pro_loss.append(loss_pro_total.item())
            
            labels = labels.round().detach().cpu().numpy().tolist()
            total_label += labels
            
            str_info = f'[Iter {i}/{len(dataloader)}]\nTotal_Pro_Loss: {round(np.mean(total_pro_loss), 4)}'
            
            if loss_func_optional is not None:         
                str_info += f' -- Con_Pro_Loss: {round(np.mean(con_pro_loss), 4)} -- Opt_Pro_Loss: {round(np.mean(opt_pro_loss), 4)}'
            
            if loss_func_cls is not None:
                str_info += f' -- Cls_Emb_Loss: {round(np.mean(cls_emb_loss), 4)}'
            
            if i % 200 == 0 or i == len(dataloader)-1:
                print(str_info)
            
                dis_info += str_info + '\n'
            
        con_pro_loss = np.mean(con_pro_loss)
        opt_pro_loss = np.mean(opt_pro_loss)
        total_pro_loss = np.mean(total_pro_loss)
        cls_emb_loss = np.mean(cls_emb_loss)
        
        loss_dict = {}
        loss_dict['total'] = total_pro_loss
        loss_dict['con'] = con_pro_loss
        loss_dict['opt'] = opt_pro_loss
        loss_dict['cls'] = cls_emb_loss
        
        dis_info += f"Pairs Dist: {len(total_label) - np.sum(total_label)} / {np.sum(total_label)}"
        
    return dis_info, loss_dict

def emb_df_combine(model, dataloader=None):
    all_ft_embed = []
    with torch.no_grad():
        model.eval()
        for i, (samples, labels) in enumerate(dataloader):
            samples = samples.to(device)
            samples_pro, samples_emb, samples_cls = model(samples)        
            samples_emb = samples_emb.cpu()
            all_ft_embed.append(samples_emb)
                
        all_ft_embed = torch.cat(all_ft_embed)
    return all_ft_embed

def classify_emb_df_combine(model, dataloader):
    all_ft_embed = []
    all_pred = []
    all_label = []
    with torch.no_grad():
        model.eval()
        
        for i, (samples, labels) in enumerate(dataloader):
            samples = samples.to(device)
            labels = labels.to(device)
            labels = labels.unsqueeze(-1)
            
            samples_pro, samples_emb, samples_cls = model(samples)
            preds_cls = sigmoid_function(samples_cls)
            
            preds = preds_cls.round().squeeze().detach().cpu().numpy().tolist()
            labels = labels.round().squeeze().detach().cpu().numpy().tolist()
            all_pred += preds
            all_label += labels
            
            samples_emb = samples_emb.cpu()
            all_ft_embed.append(samples_emb)
            
        dis_info = f"Class Samples Distribution: {len(all_label) - np.sum(all_label)} / {np.sum(all_label)}"
        bacc = balanced_accuracy_score(all_label, all_pred)
        acc = accuracy_score(all_label, all_pred)
        f1 = f1_score(all_label, all_pred)    
        all_ft_embed = torch.cat(all_ft_embed)
    return acc, bacc, f1, dis_info, all_ft_embed, all_pred, all_label
####################################################################