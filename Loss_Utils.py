import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda:0')
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from torch.autograd import Variable

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
        #scores = 1/(1+scores) # distance
        diagonal = scores.diag().view(len(images_geb), 1)
        d1 = diagonal.expand_as(scores) # direct distance
        d2 = diagonal.t().expand_as(scores) # direct distance

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin - scores + d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin - scores + d2).clamp(min=0)
        
        #print("cost_s_0")
        #print(cost_s)
        #print("cost_im_0")
        #print(cost_im)
        
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
        #print("Loss S:")
        #print(cost_s)
        #print("Loss IM:")
        #print(cost_im)
        #print("score")
        #print(scores)
        #cost_im[cost_im!=cost_im] = 1
        #cost_s[cost_s!=cost_s] = 1
        #print(cost_s.mean().item())
        #print(cost_im.mean().item())
        return cost_s.mean() + cost_im.mean()
    
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
