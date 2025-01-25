import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class CORAL_loss(nn.Module):
    def __init__(self):
        super(CORAL_loss, self).__init__()

    def forward(self, source, target):
        d = source.size(1)  

        
        source_c = self._compute_covariance_matrix(source)
        
        target_c = self._compute_covariance_matrix(target)

       
        loss = torch.mean(torch.sum((source_c - target_c) ** 2)) / (4 * d * d)
        return loss

    def _compute_covariance_matrix(self, X):
        n = X.size(0) 
        mean_X = torch.mean(X, dim=0, keepdim=True)
        X = X - mean_X
        cov_X = (X.t() @ X) / (n - 1)
        return cov_X
class MMD_loss(nn.Module):
    def __init__(self, kernel_type='linear', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd(self, X, Y):
        delta = X.mean(axis=0) - Y.mean(axis=0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            with torch.no_grad():
                XX = torch.mean(kernels[:batch_size, :batch_size])
                YY = torch.mean(kernels[batch_size:, batch_size:])
                XY = torch.mean(kernels[:batch_size, batch_size:])
                YX = torch.mean(kernels[batch_size:, :batch_size])
                loss = torch.mean(XX + YY - XY - YX)
            return loss


def _one_hot(tensor, num):
    b = list(tensor.size())[0]
    onehot = torch.cuda.FloatTensor(b, num).fill_(0)
    ones = torch.cuda.FloatTensor(b, num).fill_(1)
    out = onehot.scatter_(1, torch.unsqueeze(tensor, 1), ones)
    return out


class L1regularization(nn.Module):
    def __init__(self, weight_decay=0.1):
        super(L1regularization, self).__init__()
        self.weight_decay = weight_decay

    def forward(self, model):
        regularization_loss = 0.
        for param in model.parameters():
         # 修改

            regularization_loss += torch.mean(pow(param, 2)) * \
                self.weight_decay

        return regularization_loss


def cor(m):
    m = m.t()
    fact = 1.0 / (m.size(1) - 1)
    m = m - torch.mean(m, dim=1, keepdim=True)
    mt = m.t()
    return fact * m.matmul(mt).squeeze()


def reduction_loss(embedding, identity_matrix, size):
    loss = torch.mean(torch.abs(torch.triu(cor(embedding), diagonal=1)))
    loss = loss + 1 / torch.mean(
        torch.abs(embedding - torch.mean(embedding, dim=0).view(1, size).repeat(embedding.size()[0], 1)))
    loss = loss + torch.mean(torch.abs(embedding))
    return loss


def cosine_sim(x, y):
    x = x / torch.norm(x, dim=1, keepdim=True)
    y = y / torch.norm(y, dim=1, keepdim=True)
    sim = torch.matmul(x, torch.transpose(y, 0, 1))

    return sim


def cosine_him(x, y):
    x = x / torch.norm(x, p=2, dim=1, keepdim=True)
    y = y / torch.norm(y, p=2, dim=1, keepdim=True)
    him = torch.matmul(x, torch.transpose(y, 0, 1))
    return him


class EncodingLoss(nn.Module):
    def __init__(self, dim=64, p=0.99, use_gpu=True):
        super(EncodingLoss, self).__init__()
        if use_gpu:
            self.identity_matrix = torch.tensor(
                np.identity(dim)).float().cuda()
        else:
            self.identity_matrix = torch.tensor(np.identity(dim)).float()
        self.p = p
        self.dim = dim
        self._coral_loss = CORAL_loss()  # 可以替换为linear

    def forward(self, atac_embeddings, rna_embeddings):
        # rna
        rna_embedding_cat = rna_embeddings[0]
        rna_reduction_loss = reduction_loss(
            rna_embeddings[0], self.identity_matrix, self.dim)
        for i in range(1, len(rna_embeddings)):
            rna_embedding_cat = torch.cat(
                [rna_embedding_cat, rna_embeddings[i]], 0)
            rna_reduction_loss += reduction_loss(
                rna_embeddings[i], self.identity_matrix, self.dim)
        rna_reduction_loss /= len(rna_embeddings)

        # atac
        atac_embedding_cat = atac_embeddings[0]
        atac_reduction_loss = reduction_loss(
            atac_embeddings[0], self.identity_matrix, self.dim)
        for i in range(1, len(atac_embeddings)):
            atac_embedding_cat = torch.cat(
                [atac_embedding_cat, atac_embeddings[i]], 0)
            atac_reduction_loss += reduction_loss(
                atac_embeddings[i], self.identity_matrix, self.dim)
        atac_reduction_loss /= len(atac_embeddings)

        print(rna_embedding_cat.shape, atac_embedding_cat.shape, len(
            rna_embeddings), len(atac_embeddings))
        # cosine similarity loss
        top_k_sim = torch.topk(
             torch.max(cosine_sim(atac_embeddings[0], rna_embedding_cat), dim=1)[0],
             int(atac_embeddings[0].shape[0] * self.p))
        sim_loss = torch.mean(top_k_sim[0])

        for i in range(1, len(atac_embeddings)):
             top_k_sim = torch.topk(
                 torch.max(cosine_sim(atac_embeddings[i], rna_embedding_cat), dim=1)[0],
                 int(atac_embeddings[i].shape[0] * self.p))
             sim_loss += torch.mean(top_k_sim[0])

        sim_loss = sim_loss/len(atac_embeddings)
        # 矩阵2范数 Loss

        top_k_him = torch.topk(
            torch.max(cosine_him(
                atac_embeddings[0], rna_embedding_cat), dim=1)[0],
            int(atac_embeddings[0].shape[0] * self.p))
        him_loss = torch.mean(top_k_him[0])
        for i in range(1, len(atac_embeddings)):

            top_k_him = torch.topk(
                torch.max(cosine_him(
                    atac_embeddings[i], rna_embedding_cat), dim=1,)[0],
                int(atac_embeddings[i].shape[0] * self.p))
            him_loss += torch.mean(top_k_him[0])
        him_loss = him_loss/len(atac_embeddings)

        # mmd loss

        coral_loss = self._coral_loss(atac_embedding_cat, rna_embedding_cat)
        coral_loss = coral_loss/len(atac_embeddings) 
        loss =rna_reduction_loss + atac_reduction_loss - ((him_loss + sim_loss) / 2)+coral_loss
        #loss = rna_reduction_loss + atac_reduction_loss -((him_loss +coral_loss) / 2)
        return loss


class CenterLoss(nn.Module):
    def __init__(self, num_classes=20, feat_dim=64, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(
                self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(
                torch.randn(self.num_classes, self.feat_dim))

    def forward(self, embeddings, labels):
        center_loss = 0
        for i, x in enumerate(embeddings):
            label = labels[i].long()
            batch_size = x.size(0)
            distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(
                    self.num_classes, batch_size).t()
            distmat.addmm_(1, -2, x, self.centers.t())
            distmat = torch.sqrt(distmat)

            classes = torch.arange(self.num_classes).long()
            if self.use_gpu:
                classes = classes.cuda()
            label = label.unsqueeze(1).expand(batch_size, self.num_classes)
            mask = label.eq(classes.expand(batch_size, self.num_classes))

            dist = distmat * mask.float()
            center_loss += torch.mean(dist.clamp(min=1e-12, max=1e+12))

        center_loss = center_loss/(len(embeddings)**2)
        return center_loss


class CellLoss(nn.Module):
    def __init__(self):
        super(CellLoss, self).__init__()

    def forward(self, rna_cell_out, rna_cell_label):
        rna_cell_loss = F.cross_entropy(rna_cell_out, rna_cell_label.long())
        return rna_cell_loss
