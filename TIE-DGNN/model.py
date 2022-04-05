import datetime
import math
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from aggregator import LocalAggregator
from torch.nn import Module, Parameter
import torch.nn.functional as F
from GlobalConv_D import Globalgarph_D


class Cor_loss(Module):
    def __init__(self,cor_weight,channels,hidden_size,channel_size):
        super(Cor_loss, self).__init__()
        self.channel_num = channels
        self.cor_weight =cor_weight
        self.hidden_size = hidden_size
        self.channel_size = channel_size

    def forward(self,embedding):#解耦任意factor对
        if self.cor_weight==0:
            return 0
        else:
            embedding = embedding.view(-1, self.hidden_size)
            embedding_weight = torch.chunk(embedding, self.channel_num, dim=1)
            cor_loss = torch.tensor(0,dtype = torch.float)
            for i in range(self.channel_num):
                for j in range(i+1,self.channel_num):
                    x=embedding_weight[i]
                    y=embedding_weight[j]
                    cor_loss = cor_loss+self._create_distance_correlation(x, y)
            b=  (self.channel_num+1.0)* self.channel_num/2
            cor_loss = self.cor_weight * torch.div(cor_loss,b)
        return cor_loss

    def forward_(self,embedding):#解耦相邻factor对（内存不足情况下的次优解）
        if self.cor_weight==0:
            return 0
        else:
            embedding = embedding.view(-1,self.hidden_size)
            embedding_weight = torch.chunk(embedding, self.channel_num, dim=1)
            cor_loss = torch.tensor(0,dtype = torch.float)
            for i in range(self.channel_num-1):
                x=embedding_weight[i]
                y=embedding_weight[i+1]
                cor_loss = cor_loss+self._create_distance_correlation(x, y)
            b=  (self.channel_num+1.0)* self.channel_num/2
            cor_loss = self.cor_weight * torch.div(cor_loss,b)
        return cor_loss


    def _create_distance_correlation(self,x,y):
        zero = trans_to_cuda(torch.tensor(0,dtype=float))
        def _create_centered_distance(X,zero):
            r = torch.sum(torch.square(X),1,keepdim=True)
            X_t = torch.transpose(X,1,0)
            r_t = torch.transpose(r,1,0)
            D = torch.sqrt(torch.maximum(r-2*torch.matmul(X,X_t)+r_t,zero)+1e-8)
            D = D - torch.mean(D,dim=0,keepdim=True)-torch.mean(D,dim=1,keepdim=True)+torch.mean(D)
            return D

        def _create_distance_covariance(D1,D2,zero):
                n_samples = D1.shape[0]
                n_samples = torch.tensor(n_samples,dtype=torch.float)
                sum = torch.sum(D1*D2)
                sum = torch.div(sum,n_samples*n_samples)
                dcov=torch.sqrt(torch.maximum(sum,zero)+1e-8)
                return dcov

        D1 = _create_centered_distance(x,zero)
        D2 = _create_centered_distance(y,zero)

        dcov_12 = _create_distance_covariance(D1, D2,zero)
        dcov_11 = _create_distance_covariance(D1, D1,zero)
        dcov_22 = _create_distance_covariance(D2, D2,zero)

        dcor = torch.sqrt(torch.maximum(dcov_11 * dcov_22, zero))+1e-10
        dcor = torch.div(dcov_12,dcor)

        return dcor



class CombineGraph(Module):
    def __init__(self, opt, num_node, adj_all, num,pos_all):
        super(CombineGraph, self).__init__()
        self.opt = opt
        self.batch_size = opt.batch_size
        self.num_node = num_node
        self.dim = opt.hiddenSize
        self.dropout_local = opt.dropout_local
        self.dropout_global = opt.dropout_global
        self.hop = opt.n_iter
        self.sample_num = opt.n_sample
        self.adj_all = trans_to_cuda(torch.Tensor(adj_all)).long()
        self.num = trans_to_cuda(torch.Tensor(num)).float()
        self.pos_all=trans_to_cuda(torch.Tensor(pos_all)).float()
        self.beta=opt.conbeta
        # Aggregator
        self.local_agg = LocalAggregator(self.dim, self.opt.alpha, dropout=0.0)
        self.D_global_agg = Globalgarph_D(self.hop,opt)
        # Item representation & Position representation
        self.embedding = nn.Embedding(num_node, self.dim)
        self.pos_embedding = nn.Embedding(300, self.dim)
        self.pos_embedding_cdim = nn.Embedding(300, opt.c_dims_global[-1])
        #len
        self.hiddenSize_len=opt.hiddenSize_len
        self.len_embedding = nn.Embedding(300, self.hiddenSize_len)
        self.len_attr_transform_local = nn.Linear(self.dim+self.hiddenSize_len, self.dim, bias=True)
        self.len_attr_transform_global= nn.Linear(self.dim+self.hiddenSize_len, self.dim, bias=True)
        self.len_attr_transform=nn.Linear(self.dim, 1, bias=True)
        self.bias_list = nn.Parameter(torch.empty(size=(1, self.dim), dtype=torch.float), requires_grad=True)
        #len divide
        self.channels=opt.global_channels[-1]
        self.bias_list_cdim = nn.ParameterList(nn.Parameter(torch.empty(size=(1, opt.c_dims_global[-1]), dtype=torch.float), requires_grad=True) for i in
            range(self.channels))
        self.len_attr_transform_list=nn.ParameterList(
            nn.Parameter(torch.empty(size=(opt.c_dims_global[-1]+self.hiddenSize_len, 1), dtype=torch.float), requires_grad=True) for i in
            range(self.channels))
        self.w_11 = nn.Parameter(torch.Tensor(2 * opt.c_dims_global[-1], opt.c_dims_global[-1]))
        self.w_22 = nn.Parameter(torch.Tensor(opt.c_dims_global[-1], 1))
        self.glu11 = nn.Linear(opt.c_dims_global[-1], opt.c_dims_global[-1])
        self.glu22 = nn.Linear(opt.c_dims_global[-1], opt.c_dims_global[-1], bias=False)
        self.dim_tran_cdim=nn.Linear(self.dim,opt.c_dims_global[-1])

        #pos
        self.pos_before=nn.Embedding(10, opt.hiddenSize_pos)
        self.pos_after = nn.Embedding(10, opt.hiddenSize_pos)
        self.pos_io = nn.Embedding(2, opt.hiddenSize_pos)
        # Parameters
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.glu1 = nn.Linear(self.dim, self.dim)
        self.glu2 = nn.Linear(self.dim, self.dim, bias=False)
        self.linear_transform = nn.Linear(self.dim, self.dim, bias=False)


        self.leakyrelu = nn.LeakyReLU(opt.alpha)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def sample_all(self, target):
        adj_all=[]
        num_all=[]
        pos_all=[]
        b=len(target)
        for j in range(len(target)):
            adj_in=self.adj_all[0][target[j].view(-1)]
            num_in=self.num[0][target[j].view(-1)]
            pos_in=self.pos_all[0][target[j].view(-1)]
            adj_out=self.adj_all[1][target[j].view(-1)]
            num_out=self.num[1][target[j].view(-1)]
            pos_out=self.pos_all[1][target[j].view(-1)]
            adj_io = self.adj_all[2][target[j].view(-1)]
            num_io = self.num[2][target[j].view(-1)]
            adj_all.append(adj_in)
            adj_all.append(adj_out)
            adj_all.append(adj_io)
            num_all.append(num_in)
            num_all.append(num_out)
            num_all.append(num_io)
            pos_all.append(pos_in)
            pos_all.append(pos_out)
            pos_all.append(pos_out)

        return adj_all,num_all,pos_all


    def SSL(self, sess_emb_hgnn, sess_emb_lgcn):

        def row_column_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            corrupted_embedding = corrupted_embedding[:,torch.randperm(corrupted_embedding.size()[1])]
            return corrupted_embedding
        def score(x1, x2):
            return torch.sum(torch.mul(x1, x2), 1)

        pos = trans_to_cuda(score(sess_emb_hgnn, sess_emb_lgcn))
        neg1 = trans_to_cuda(score(sess_emb_lgcn, row_column_shuffle(sess_emb_hgnn)))
        one  = trans_to_cuda(torch.ones(neg1.shape[0]))
        con_loss = trans_to_cuda(torch.sum(-torch.log(1e-8 + torch.sigmoid(pos))-torch.log(1e-8 + (one - torch.sigmoid(neg1)))))
        return self.beta*con_loss



    def compute_scores_divide_(self, local_hidden,global_hidden, mask,lendata):
        #local
        local_hidden = F.dropout(local_hidden, self.dropout_local, training=self.training)
        mask = mask.float().unsqueeze(-1)

        batch_size = local_hidden.shape[0]
        len_local = local_hidden.shape[1]
        pos_emb = self.pos_embedding.weight[:len_local]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)

        hs = torch.sum(local_hidden * mask, -2) / torch.sum(mask, 1)
        hs = hs.unsqueeze(-2).repeat(1, len_local, 1)
        nh = torch.matmul(torch.cat([pos_emb, local_hidden], -1), self.w_1)
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        h_local = torch.sum(beta * local_hidden, 1)

        #global
        global_hidden = F.dropout(global_hidden, self.dropout_global, training=self.training)

        global_hidden_channel = torch.chunk(global_hidden, self.channels, dim=2)

        len_ = global_hidden_channel[0].shape[1]
        pos_emb = self.pos_embedding_cdim.weight[:len_]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)
        session_infos=[]
        for global_seq in global_hidden_channel:
            hs = torch.sum(global_seq * mask, -2) / torch.sum(mask, 1)
            hs = hs.unsqueeze(-2).repeat(1, len_, 1)
            nh = torch.matmul(torch.cat([pos_emb, global_seq], -1), self.w_11)
            nh = torch.tanh(nh)
            nh = torch.sigmoid(self.glu11(nh) + self.glu22(hs))
            beta = torch.matmul(nh, self.w_22)
            beta = beta * mask
            h_global = torch.sum(beta * global_seq, 1)
            session_infos.append(h_global)
        session_infos = torch.cat([session_infos[i] for i in range(len(session_infos))], dim=1)
        b=self.embedding.weight[1:]
        len_emb = torch.cat([self.len_embedding(lendata[i]).unsqueeze(0) for i in range(len(lendata))], dim=0)
        conloss = self.SSL(h_local, session_infos)
        output=h_local+session_infos+self.bias_list
        #output = h_local + session_infos
        score = torch.matmul(output, b.transpose(1, 0))
        return score,conloss


    def forward_divide(self, inputs, adj, mask_item, item,lendata,alias_inputs):#inputs图上的点 items是session中的序列点
        batch_size = inputs.shape[0]
        seqs_len = inputs.shape[1]
        h = self.embedding(inputs)
        # local
        h_local = self.local_agg(h, adj, mask_item)
        # len emb
        len_emb = torch.cat([self.len_embedding(lendata[i]).unsqueeze(0) for i in range(len(lendata))], dim=0)
        len_emb = len_emb.unsqueeze(1).repeat(1, h_local.size(1), 1)
        # get local session emb
        alias_inputs = alias_inputs
        get = lambda index: h_local[index][alias_inputs[index]]
        seq_hidden_local = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
        # global
        item_neighbors = [[inputs]]
        weight_neighbors = []
        pos_neighbors = []
        support_size = seqs_len
        for i in range(1, self.hop + 1):
            item_sample_i, weight_sample_i, pos_sample_i = self.sample_all(item_neighbors[i - 1])
            support_size *= self.sample_num
            item_neighbors.append([item_sample_i[j].view(batch_size, support_size) for j in range(len(item_sample_i))])
            weight_neighbors.append(
                [weight_sample_i[j].view(batch_size, support_size) for j in range(len(weight_sample_i))])
            pos_neighbors.append([pos_sample_i[j].view(batch_size, support_size) for j in range(len(pos_sample_i))])
        h_cor = [self.embedding(i) for i in item_neighbors[0]][0]
        cor_hidden = self.D_global_agg.rout_emb_cor(h_cor)  # 配cor
        h_global = self.D_global_agg(item_neighbors, self.embedding, weight_neighbors, seq_hidden_local, mask_item,
                                     pos_neighbors, self.pos_before,
                                     self.pos_after,self.pos_io)
        # combine
        output=[h_local,h_global]
        return output,cor_hidden






def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable



def forward_divide(model, data):
    alias_inputs, adj, items, mask, targets, inputs , lendata = data
    alias_inputs = trans_to_cuda(alias_inputs).long()
    items = trans_to_cuda(items).long()
    adj = trans_to_cuda(adj).float()
    mask = trans_to_cuda(mask).long()
    lendata = trans_to_cuda(lendata).long()
    inputs = trans_to_cuda(inputs).long()
    hidden,cor_hidden = model.forward_divide(items, adj, mask, inputs,lendata,alias_inputs)
    get_local = lambda index: hidden[0][index][alias_inputs[index]]
    seq_hidden_local = torch.stack([get_local(i) for i in torch.arange(len(alias_inputs)).long()])
    get_global = lambda index: hidden[1][index][alias_inputs[index]]
    seq_hidden_global = torch.stack([get_global(i) for i in torch.arange(len(alias_inputs)).long()])
    scores,conloss=model.compute_scores_divide_(seq_hidden_local,seq_hidden_global, mask,lendata)
    return targets, scores,conloss,cor_hidden




def train_test(model, train_data, test_data,Cor_loss_model):
    print('start training: ', datetime.datetime.now())
    model.train()
    Cor_loss_model.train()
    total_loss = 0.0
    total_corloss=0.0
    total_conloss=0.0
    train_loader = torch.utils.data.DataLoader(train_data, num_workers=4, batch_size=model.batch_size,
                                               shuffle=True, pin_memory=True)
    for data in tqdm(train_loader):
        model.optimizer.zero_grad()

        targets, scores,conloss, cor_hidden = forward_divide(model, data)
        targets = trans_to_cuda(targets).long()
        loss = model.loss_function(scores, targets - 1)
        cor_loss=Cor_loss_model(cor_hidden)

        loss=loss+cor_loss+conloss

        loss.backward()
        model.optimizer.step()
        total_loss += loss
        total_corloss+=cor_loss
        total_conloss += conloss
    print('\tLoss:\t%.3f' % total_loss)
    print('\tcorLoss:\t%.3f' % total_corloss)
    print('\tconloss:\t%.3f' % total_conloss)
    model.scheduler.step()
    top_K = [5,10,20]
    metrics = {}
    for K in top_K:
        metrics['hit%d' % K] = []
        metrics['mrr%d' % K] = []
    print('start predicting: ', datetime.datetime.now())
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size=model.batch_size,
                                              shuffle=False, pin_memory=True)

    for data in test_loader:
        targets, scores ,_,_= forward_divide(model, data)
        targets = targets.numpy()
        for i in top_K:
            sub_scores = scores.topk(i)[1]
            sub_scores = trans_to_cpu(sub_scores).detach().numpy()
            for score, target, mask in zip(sub_scores, targets, test_data.mask):
                metrics['hit%d' % i].append(np.isin(target - 1, score))
                if len(np.where(score == target - 1)[0]) == 0:
                    metrics['mrr%d' % i].append(0)
                else:
                    metrics['mrr%d' % i].append(1 / (np.where(score == target - 1)[0][0] + 1))

    for K in top_K:
        metrics['hit%d' % K] = np.mean(metrics['hit%d' % K]) * 100
        metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K]) * 100


    return metrics
