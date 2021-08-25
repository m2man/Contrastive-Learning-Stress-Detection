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

def logging(s, path, print_=False):
    if print_:
        print(s)
    if path:
        with open(path, 'a+') as f:
            f.write(s + '\n')
            
# Loss function    
class TripletLoss(nn.Module):
    """
    Triplet loss function.
    """

    def __init__(self, margin=2.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):

        squarred_distance_1 = (anchor - positive).pow(2).sum(1).pow(1/2)
        
        squarred_distance_2 = (anchor - negative).pow(2).sum(1).pow(1/2)
        
        triplet_loss = F.relu(self.margin + squarred_distance_1 - squarred_distance_2).mean()
        
        return triplet_loss

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
        d1 = diagonal.expand_as(scores) # direct distance
        d2 = diagonal.t().expand_as(scores) # direct distance

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
    
class ContrastiveLoss_EuclidSimilarity(nn.Module):
    def __init__(self, margin=2, max_violation=False):
        super(ContrastiveLoss_EuclidSimilarity, self).__init__()
        self.max_violation = max_violation
        self.margin = margin
        
    def forward(self, images_geb, captions_geb):
        scores = torch.cdist(images_geb, captions_geb, p=2) # nimage x ncaption
        diagonal = scores.diag().view(len(images_geb), 1)
        d1 = diagonal.expand_as(scores) # direct distance
        d2 = diagonal.t().expand_as(scores) # direct distance

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin - scores + d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin - scores + d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

#         cost_s = torch.where(torch.isnan(cost_s), torch.zeros_like(cost_s), cost_s)
#         cost_s = torch.where(torch.isinf(cost_s), torch.zeros_like(cost_s), cost_s)
#         cost_im = torch.where(torch.isnan(cost_im), torch.zeros_like(cost_im), cost_im)
#         cost_im = torch.where(torch.isinf(cost_im), torch.zeros_like(cost_im), cost_im)
#         cost_s = torch.clamp(cost_s, 0, 2*self.margin)
#         cost_im = torch.clamp(cost_im, 0, 2*self.margin)

#         if torch.isnan(cost_s).any():
#             print('cost_s co NaN')
#             print(cost_s)
#             print(d1)
#         if torch.isnan(cost_im).any():
#             print('cost_im co NaN')
#             print(cost_im)
#             print(d2)
        
        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        #cost_im[cost_im!=cost_im] = 1
        #cost_s[cost_s!=cost_s] = 1
        #print(cost_s.mean().item())
        #print(cost_im.mean().item())
        return cost_s.mean() + cost_im.mean()
    
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