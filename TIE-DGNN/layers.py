import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from aggregator import  GlobalAggregator_io

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

class Conv(nn.Module):
    def __init__(self, in_dim,channels,c_dims_global,hop,opt):
        super(Conv, self).__init__()
        self.in_dim = in_dim
        self.hop = hop
        self.channels = channels
        self.c_dim=c_dims_global

        self.indim_tran_cdim = nn.Linear(self.in_dim, self.c_dim, bias=True)

        self.sample_num = opt.n_sample
        self.weight_list = nn.ParameterList(
            nn.Parameter(torch.empty(size=(self.in_dim, self.c_dim), dtype=torch.float), requires_grad=True) for i in
            range(self.channels))
        self.bias_list = nn.ParameterList(
            nn.Parameter(torch.empty(size=(1, self.c_dim), dtype=torch.float), requires_grad=True) for i in
            range(self.channels))
        self.linear_1 = nn.Linear(self.in_dim, self.in_dim, bias=True)
        self.linear_2 = nn.Linear(self.in_dim, self.in_dim, bias=True)
        self.linear_3 = nn.Linear(self.in_dim, 1, bias=False)
        self.hiddenSize_pos=opt.hiddenSize_pos
        #global
        self.global_agg = []
        for i in range(self.hop):
            if opt.activate == 'relu':
                agg = GlobalAggregator_io(self.c_dim, opt.dropout_gcn, self.hiddenSize_pos,act=torch.relu)
            else:
                agg = GlobalAggregator_io(self.c_dim, opt.dropout_gcn, self.hiddenSize_pos,act=torch.tanh,)
            self.add_module('agg_gcn_{}'.format(i), agg)
            self.global_agg.append(agg)






    def forward(self, h, item_neighbors,embedding,weight_neighbors,seq_hidden_local,mask_item,pos_neighbors,pos_before,pos_after):
        batchsize = h.shape[0]

        entity_vectors_ = []
        for j in range(len(item_neighbors)):
            entity_vectors_.append([embedding(i) for i in item_neighbors[j]])  # 第一个是本session图中点的的表示 第二个列表是图中点对应的邻居节点表示

        entity_vectors_[0][0] = h


        weight_vectors = weight_neighbors

        pos_vectors = pos_neighbors




        entity_vectors = []
        for i in range(len(entity_vectors_)):
            temp=[]
            for j in range(len(entity_vectors_[i])):
                entity_vectors1 = self.route_emb_global(entity_vectors_[i][j])
                temp.append(entity_vectors1)
            entity_vectors.append(temp)


        entity_vectors_temp = []
        for j in range(len(entity_vectors[0][0])):
            entity_vectors_temp1 = []
            for i in range(len(entity_vectors)):
                entity_vectors_temp2=[]
                for k in range(len(entity_vectors[i])):
                    entity_vectors_temp2.append(entity_vectors[i][k][j])
                entity_vectors_temp1.append(entity_vectors_temp2)
            entity_vectors_temp.append(entity_vectors_temp1)
        entity_vectors=entity_vectors_temp



        # 解纠缠平均会话表示
        session_info = []
        item_emb = self.route_emb_global(seq_hidden_local)
        for i in range(len(item_emb)):
            sum_item_emb_ = torch.sum(item_emb[i], 1) / torch.sum(mask_item.float(), -1).unsqueeze(-1)
            sum_item_emb_ = sum_item_emb_.unsqueeze(-2)
            sum_item_emb__ = []
            for j in range(self.hop):
                sum_item_emb = sum_item_emb_.repeat(1, entity_vectors[i][j][0].shape[1], 1)
                sum_item_emb__.append(sum_item_emb)
            session_info.append(sum_item_emb__)

        for j in range(len(entity_vectors)):
            for n_hop in range(self.hop):
                entity_vectors_next_iter = []
                shape = [batchsize, -1, self.sample_num, self.c_dim]
                for hop in range(self.hop - n_hop):
                    aggregator = self.global_agg[n_hop]
                    vector = aggregator(self_vectors=entity_vectors[j][hop],
                                        neighbor_vector=[entity_vectors[j][hop + 1][k].view(shape) for k in range(len(entity_vectors[j][hop + 1]))],
                                        masks=None,
                                        batch_size=batchsize,
                                        neighbor_weight=[weight_vectors[hop][k].view(batchsize, -1, self.sample_num) for k in range(len(weight_vectors[hop]))],
                                        pos_weight=[pos_vectors[hop][k].view(batchsize, -1, self.sample_num,self.hiddenSize_pos) for k in range(len(pos_vectors[hop]))],
                                        extra_vector=session_info[j][hop])
                    entity_vectors_next_iter.append(vector)
                entity_vectors[j] = entity_vectors_next_iter
        h_global = entity_vectors
        h_global = torch.cat([h_global[i][0][0] for i in range(len(h_global))], dim=2)
        q1 = self.linear_1(h_global)
        q2 = self.linear_2(h)
        alpha = self.linear_3(torch.sigmoid(q1 + q2))
        h_global = alpha * h + (1 - alpha) * h_global
        return h_global




    def route_emb_global(self,embedding):
        c_features = []
        for i in range(self.channels): 
            # z = torch.mm(features, self.weight_list[i]) + self.bias_list[i]
            z = torch.matmul(embedding, self.weight_list[i]) + self.bias_list[i]
            z = F.normalize(z, dim=2)
            c_features.append(z)
        return c_features

    def route_embedding(self,embedding):
        c_features = []
        for i in range(self.channels): 
            z = torch.matmul(embedding, self.weight_list[i]) + self.bias_list[i]
            z = F.normalize(z, dim=2)
            c_features.append(z)
        return c_features

