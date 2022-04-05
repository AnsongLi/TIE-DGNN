import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
from layers import *

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

class Globalgarph_D(nn.Module):

    def __init__(self,hop,opt):
        super(Globalgarph_D, self).__init__()
        self.c_dims_global=opt.c_dims_global
        self.in_dim=opt.in_dims
        self.hop=hop
        self.dropout=opt.DG_dropout
        self.channels=opt.global_channels
        D_conv=[]
        for x in range(len(self.channels)):
            G_conv = Conv(self.in_dim[x], self.channels[x], self.c_dims_global[x], self.hop,opt)
            D_conv.append(G_conv)
        self.D_conv = nn.Sequential(*D_conv)

    def forward(self,item_neighbors,embedding,weight_neighbors,seq_hidden_local,mask_item,pos_neighbors,pos_before,pos_after,pos_io):
        h  = [ embedding(i) for i in item_neighbors[0]][0]
        for j in range(len(pos_neighbors)):
            for i in range(0,len(pos_neighbors[j]),3):
                pos_neighbors[j][i]= pos_before(pos_neighbors[j][i].long())

            for k in range(1,len(pos_neighbors[j]),3):
                pos_neighbors[j][k] = pos_after(pos_neighbors[j][k].long())

            for z in range(2,len(pos_neighbors[j]),3):
                pos_neighbors[j][z]=pos_io(trans_to_cuda(torch.tensor([1]))).unsqueeze(0).repeat(pos_neighbors[j][k].shape[0],pos_neighbors[j][k].shape[1],1)
        for i in range(len(self.D_conv)-1):
            h = self.D_conv[i](h,item_neighbors,embedding,weight_neighbors,seq_hidden_local,mask_item,pos_neighbors,pos_before,pos_after)
            h = F.dropout(h, self.dropout, training=self.training)
        h = self.D_conv[len(self.D_conv)-1](h,item_neighbors,embedding,weight_neighbors,seq_hidden_local,mask_item,pos_neighbors,pos_before,pos_after)
        return h





    def rout_emb_cor(self,emb):
        cor_hidden=trans_to_cuda(torch.tensor([]))
        for i in range(len(self.D_conv[0].route_embedding(emb))):
            cor_hidden=torch.cat((cor_hidden,self.D_conv[0].route_embedding(emb)[i]),2)

        return cor_hidden

