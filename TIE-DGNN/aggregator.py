import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import numpy


class LocalAggregator(nn.Module):
    def __init__(self, dim, alpha, dropout=0., name=None):
        super(LocalAggregator, self).__init__()
        self.dim = dim
        self.dropout = dropout

        self.a_0 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_1 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_3 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.bias = nn.Parameter(torch.Tensor(self.dim))

        self.leakyrelu = nn.LeakyReLU(alpha)


    def forward(self, hidden, adj, mask_item=None):
        h = hidden
        batch_size = h.shape[0]
        N = h.shape[1]

        a_input = (h.repeat(1, 1, N).view(batch_size, N * N, self.dim)
                   * h.repeat(1, N, 1)).view(batch_size, N, N, self.dim)

        e_0 = torch.matmul(a_input, self.a_0)
        e_1 = torch.matmul(a_input, self.a_1)
        e_2 = torch.matmul(a_input, self.a_2)
        e_3 = torch.matmul(a_input, self.a_3)

        e_0 = self.leakyrelu(e_0).squeeze(-1).view(batch_size, N, N)
        e_1 = self.leakyrelu(e_1).squeeze(-1).view(batch_size, N, N)
        e_2 = self.leakyrelu(e_2).squeeze(-1).view(batch_size, N, N)
        e_3 = self.leakyrelu(e_3).squeeze(-1).view(batch_size, N, N)

        mask = -9e15 * torch.ones_like(e_0)
        alpha = torch.where(adj.eq(1), e_0, mask)
        alpha = torch.where(adj.eq(2), e_1, alpha)
        alpha = torch.where(adj.eq(3), e_2, alpha)
        alpha = torch.where(adj.eq(4), e_3, alpha)
        alpha = torch.softmax(alpha, dim=-1)

        output = torch.matmul(alpha, h)
        return output


class GlobalAggregator_io(nn.Module):
    def __init__(self, dim, dropout, hiddenSize_pos,act=torch.relu, name=None):
        super(GlobalAggregator_io, self).__init__()
        self.dropout = dropout
        self.act = act
        self.dim = dim

        self.w_11 = nn.Parameter(torch.Tensor(self.dim + 1+hiddenSize_pos, self.dim))
        self.w_12 = nn.Parameter(torch.Tensor(self.dim + 1+hiddenSize_pos, self.dim))
        self.w_13 = nn.Parameter(torch.Tensor(self.dim + 1 + hiddenSize_pos, self.dim))
        self.w_21 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.w_22 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.w_23 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.w_31 = nn.Parameter(torch.Tensor(3 * self.dim, self.dim))
        self.w_32 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.bias = nn.Parameter(torch.Tensor(self.dim))
        
    

    def forward(self, self_vectors, neighbor_vector, batch_size, masks, neighbor_weight, pos_weight,extra_vector=None):
        h=[]
        for i in range(len(self_vectors)):
            if extra_vector is not None:
                alpha = torch.matmul(
                                    torch.cat([extra_vector.unsqueeze(2).repeat(1, 1, neighbor_vector[(i+1)*3-3].shape[2], 1)*neighbor_vector[(i+1)*3-3], neighbor_weight[(i+1)*3-3].unsqueeze(-1),pos_weight[(i+1)*3-3]], -1),
                                     self.w_11).squeeze(-1)
                alpha = F.leaky_relu(alpha, negative_slope=0.2)
                alpha = torch.matmul(alpha, self.w_21).squeeze(-1)
                alpha = torch.softmax(alpha, -1).unsqueeze(-1)
                neighbor_vector_in = torch.sum(alpha * neighbor_vector[(i+1)*3-3], dim=-2)

                beta = torch.matmul(
                                    torch.cat([extra_vector.unsqueeze(2).repeat(1, 1, neighbor_vector[(i+1)*3-2].shape[2], 1)*neighbor_vector[(i+1)*3-2], neighbor_weight[(i+1)*3-2].unsqueeze(-1),pos_weight[(i+1)*3-2]], -1),
                                     self.w_12).squeeze(-1)
                beta = F.leaky_relu(beta, negative_slope=0.2)
                beta = torch.matmul(beta, self.w_22).squeeze(-1)
                beta = torch.softmax(beta, -1).unsqueeze(-1)
                neighbor_vector_out = torch.sum(beta * neighbor_vector[(i+1)*3-2], dim=-2)

                thea = torch.matmul(
                                    torch.cat([extra_vector.unsqueeze(2).repeat(1, 1, neighbor_vector[(i+1)*3-1].shape[2], 1)*neighbor_vector[(i+1)*3-1], neighbor_weight[(i+1)*3-1].unsqueeze(-1),pos_weight[(i+1)*3-1]], -1),
                                     self.w_13).squeeze(-1)
                thea = F.leaky_relu(thea, negative_slope=0.2)
                thea = torch.matmul(thea, self.w_23).squeeze(-1)
                thea = torch.softmax(thea, -1).unsqueeze(-1)
                neighbor_vector_io = torch.sum(thea * neighbor_vector[(i+1)*3-1], dim=-2)


                neighbor_vector_=neighbor_vector_in+neighbor_vector_out+neighbor_vector_io
            else:
                neighbor_vector_ = torch.mean(neighbor_vector_, dim=2)
   
            output = torch.cat([self_vectors[i], neighbor_vector_], -1)
            output = F.dropout(output, self.dropout, training=self.training)
            output = torch.matmul(output, self.w_32)
            output = output.view(batch_size, -1, self.dim)
            output = self.act(output)
            h.append(output)

        return h

