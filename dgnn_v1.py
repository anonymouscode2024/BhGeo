import torch
from torch import nn
from dgl.nn.pytorch import GraphConv,NNConv
import os
import numpy as np
import dgl
import random as rd
import torch
import torch.nn as nn


class MPNN(nn.Module):
    def __init__(self, aggregator_type,node_in_feats, node_hidden_dim, edge_input_dim, edge_hidden_dim,num_step_message_passing,gconv_dp,edge_dp,nn_dp1,alpha,beta):
     
        super(MPNN, self).__init__()
        self.lin0 = nn.Linear(node_in_feats, node_hidden_dim)#65,32
        self.num_step_message_passing=num_step_message_passing#
        edge_network = nn.Sequential(
         
            nn.Linear(edge_input_dim, edge_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=edge_dp),
            nn.Linear(edge_hidden_dim, node_hidden_dim * node_hidden_dim),
            nn.Dropout(p=edge_dp))#1-4-32x32

        self.conv = NNConv(in_feats=node_hidden_dim,#32
               out_feats=node_hidden_dim,#32
               edge_func=edge_network,#32x32
               aggregator_type=aggregator_type,
                           residual=True)
        self.lin1 = nn.Linear(node_hidden_dim, node_hidden_dim)  # 65,32


        self.y_linear = nn.Linear(node_hidden_dim, 3)# lat lon delay
        self.bn = nn.BatchNorm1d(node_hidden_dim)

        self.y_linear2 = nn.Linear(node_hidden_dim, 1)  # 4-4

        self.alpha = alpha
        self.beta = beta
        self.gnn_dropout = nn.Dropout(p=gconv_dp)#dropout
        self.nn_dropout = nn.Dropout(p=nn_dp1)
        # self.nn_dropout2 = nn.Dropout(p=nn_dp2)

    def forward(self, g, n_feat, e_feat,src_list,dst_list):


        out = torch.relu(self.lin0(n_feat))  # (B1, H1)
        h0= out.clone()
        self.beta = 1.0/self.num_step_message_passing
        for i in range(self.num_step_message_passing):
            temp=self.alpha * self.conv(g, out, e_feat) + (1 - self.alpha) * h0
            out = torch.relu(self.beta*self.lin1(temp)+(1-self.beta)*temp)
            out = self.gnn_dropout(out)# (B1, H1)

        #edge hop estimation
        y_bn = self.bn(out)
        edge_hop_out=torch.sigmoid(self.y_linear2(y_bn[src_list]*y_bn[dst_list]))#
        y_sigmoid = torch.sigmoid(self.y_linear(y_bn))#

        # y_bn = self.bn(self.y_linear(out))
        # y_sigmoid = torch.sigmoid(y_bn)

        #RuntimeError: mat1 and mat2 shapes cannot be multiplied (2878x32 and 4x4)
        # y2_relu = torch.sigmoid(self.y2_linear(out))
        # y2_relu_dp = self.nn_dropout2(y2_relu)
        # y1_predict = self.y1_predict(y1_relu_dp)
        # y2_predict = self.y2_predict(y2_relu_dp)

        return [y_sigmoid,edge_hop_out]
