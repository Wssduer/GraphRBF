import sys

import torch
from torch import nn
from torch_cluster import knn_graph, radius_graph
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_max
import numpy as np

from sklearn.mixture import GaussianMixture


class AttentionModule(nn.Module):
    def __init__(self, input_size):
        super(AttentionModule, self).__init__()
        self.attention_weights = torch.nn.Parameter(torch.rand(input_size))

    def forward(self, x):
        attention_scores = F.softmax(self.attention_weights, dim=0)
        attended_features = x * attention_scores
        output = attended_features
        return output


class AttentionAggregation(nn.Module):
    def __init__(self, input_size, num_centers):
        super(AttentionAggregation, self).__init__()
        self.num_centers = num_centers
        self.W = nn.Linear(input_size, 1)

    def forward(self, input_tensor):
        w = self.W(input_tensor)
        attention_weights = F.softmax(w, dim=1) 
        aggregated_feature = torch.sum(attention_weights * input_tensor, dim=1, keepdim=True) 

        return aggregated_feature


class ENPGModel(torch.nn.Module):
    def __init__(self, x_ind, x_hs, edge_ind, e_hs, u_ind, hs, pos_ind, p_hs, dropratio, bias=True):
        super(ENPGModel, self).__init__()
        self.pos_ind = pos_ind

        Eind = x_ind * 2 + edge_ind + u_ind + pos_ind * 3
        self.Emlp = nn.Sequential(
            nn.Linear(Eind, e_hs, bias=bias),
            nn.BatchNorm1d(e_hs),
            nn.ReLU(inplace=True),
            nn.Dropout(dropratio),
            nn.Linear(e_hs, e_hs, bias=bias))

        Nind = x_ind + hs + u_ind + pos_ind
        self.Nmlp = nn.Sequential(
            nn.Linear(Nind, x_hs, bias=bias),
            nn.BatchNorm1d(x_hs),
            nn.ReLU(inplace=True),
            nn.Dropout(dropratio),
            nn.Linear(x_hs, x_hs, bias=bias),
            nn.BatchNorm1d(hs))

        Gind = u_ind + hs * 2
        self.Gmlp = nn.Sequential(
            nn.Linear(Gind, hs, bias=bias),
            nn.BatchNorm1d(hs),
            nn.ReLU(inplace=True),
            nn.Dropout(dropratio),
            nn.Linear(hs, hs, bias=bias))

        self.Pmlp = nn.Sequential(
            nn.Linear(pos_ind, p_hs, bias=bias),
            nn.BatchNorm1d(p_hs),
            nn.ReLU(inplace=True),
            nn.Dropout(dropratio),
            nn.Linear(p_hs, p_hs, bias=bias))

        self.EPmlp = nn.Sequential(
            nn.Linear(pos_ind + 3, p_hs, bias=bias),
            nn.BatchNorm1d(p_hs),
            nn.ReLU(inplace=True),
            nn.Dropout(dropratio),
            nn.Linear(p_hs, p_hs, bias=bias))

        self.RBFBlock = EnhancedRBFModel(hidden_size=hs, num_kernels=32, n_filters=64, output_dim=hs,
                                         dropratio=dropratio, bias=bias)
        self.RBFABlock = RBFAModel(hidden_size=hs, num_kernels=32, n_filters=64, output_dim=hs,
                                   dropratio=dropratio, bias=bias)
        self.Nattention = AttentionModule(Nind)
        self.Eattention = AttentionModule(Eind)
        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.Nattention]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self, x, edge_index, edge_attr, pos, posemb, u, batch):
        row, col = edge_index
        u_out = []
        n_out = [x, posemb]
        e_out = [x[row], x[col], edge_attr, posemb[row], posemb[col], posemb[col] - posemb[row]]
        if u is not None:
            n_out.append(u[batch])
            e_out.append(u[batch[row]])
            u_out.append(u)

        e_out = torch.cat(e_out, dim=-1)
        e_out = self.Emlp(e_out)
        n_out.append(scatter_add(e_out, col, dim=0, dim_size=x.size(0)))
        n_out = torch.cat(n_out, dim=1)
        n_out = self.Nmlp(n_out)
        if self.pos_ind == 3:
            posemb = self.Pmlp(pos)
        else:
            posemb = self.EPmlp(torch.cat([posemb, pos], dim=1))

        # Graph feature extract
        u = self.RBFABlock(n_out, pos, batch)
        u_out.append(u)
        u_out.append(scatter_add(n_out, batch, dim=0))
        u_out = torch.cat(u_out, dim=1)
        u_out = self.Gmlp(u_out)

        return n_out, e_out, u_out, posemb


class PosEmb(torch.nn.Module):
    def __init__(self, pos_ind, p_hs, dropratio, bias=True):
        super(PosEmb, self).__init__()
        self.pos_encoder = nn.Sequential(
            nn.Conv1d(pos_ind, p_hs, kernel_size=1, bias=bias),
            nn.BatchNorm1d(p_hs),
            nn.ReLU(inplace=True),
            nn.Dropout(dropratio),
            nn.Conv1d(p_hs, p_hs, kernel_size=1, bias=bias),
            nn.BatchNorm1d(p_hs))

    def forward(self, pos):
        pos = pos.permute(1, 0).unsqueeze(0)
        out = self.pos_encoder(pos).squeeze().permute(1, 0)
        return out


class MultiPosEmb(torch.nn.Module):
    def __init__(self, pos_ind, p_hs, dropratio, bias=True):
        super(MultiPosEmb, self).__init__()
        self.multi_pos_encoder = nn.Sequential(
            nn.Conv1d(pos_ind * 3, p_hs, kernel_size=1, bias=bias),
            nn.BatchNorm1d(p_hs),
            nn.ReLU(inplace=True),
            nn.Dropout(dropratio),
            nn.Conv1d(p_hs, p_hs, kernel_size=1, bias=bias),
            nn.BatchNorm1d(p_hs))

    def forward(self, pos1, pos2):
        out = torch.cat([pos1, pos2, pos1 - pos2], dim=1)
        out = out.permute(1, 0).unsqueeze(0)
        out = self.multi_pos_encoder(out).squeeze().permute(1, 0)
        return out


class RBFNN(nn.Module):
    def __init__(self, input_dim, num_centers):
        super(RBFNN, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_centers, 3))
        self.widths = nn.Parameter(torch.ones(num_centers))
        self.ReLU = nn.LeakyReLU(0.2)
        self.x1 = nn.Linear(input_dim, 1)
        self.x2 = nn.Linear(num_centers, 1)
        self.pos = nn.Linear(num_centers, 1)
        self.gaussian_kernel = GaussianKernel(input_dim=3, num_centers=num_centers)

    def gaussian_rbf(self, pos):
        distances = torch.sqrt(torch.sum((pos.unsqueeze(1) - self.centers) ** 2, dim=2))
        return torch.exp(-distances / (2 * self.widths ** 2))

    def multiquadric_rbf(self, pos):
        distances = torch.sqrt(torch.sum((pos.unsqueeze(1) - self.centers) ** 2, dim=2))
        return torch.sqrt(1 + (distances / self.widths) ** 2)

    def inv_multiquadric_rbf(self, pos):
        distances = torch.sqrt(torch.sum((pos.unsqueeze(1) - self.centers) ** 2, dim=2))
        return 1 / torch.sqrt(1 + (distances / self.widths) ** 2)

    def forward(self, x, pos):
        x1 = self.x1(x)
        rbf_output = self.gaussian_rbf(pos) 
        mul_output = torch.mul(rbf_output, x1) 
        x2 = self.x2(mul_output + rbf_output) 
        pos1 = self.pos(rbf_output) 
        output = self.ReLU(x2 + pos1)  
        return output


class RBFANN(nn.Module):
    def __init__(self, input_dim, hs, num_centers):
        super(RBFANN, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_centers, 3))
        self.widths = nn.Parameter(torch.ones(num_centers))
        self.ReLU = nn.LeakyReLU(0.2)
        self.attention = AttentionAggregation(hs, num_centers)
        self.gaussian_kernel = GaussianKernel(input_dim=3, num_centers=num_centers)
        self.x1 = nn.Linear(input_dim, hs)
        self.x2 = nn.Linear(hs, input_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.x3 = nn.Linear(input_dim, 1)

    def gaussian_rbf(self, pos):
        distances = torch.sqrt(torch.sum((pos.unsqueeze(1) - self.centers) ** 2, dim=2))
        return torch.exp(-distances / (self.widths ** 2))

    def multiquadric_rbf(self, pos):
        distances = torch.sqrt(torch.sum((pos.unsqueeze(1) - self.centers) ** 2, dim=2))
        return torch.sqrt(1 + (distances / self.widths) ** 2)

    def inv_multiquadric_rbf(self, pos):
        distances = torch.sqrt(torch.sum((pos.unsqueeze(1) - self.centers) ** 2, dim=2))
        return 1 / torch.sqrt(1 + (distances / self.widths) ** 2)

    def forward(self, x, pos, batch):
        x1 = self.x1(x)
        rbf_output = self.gaussian_rbf(pos)
        mul_output = rbf_output.unsqueeze(2) * x1.unsqueeze(1)
        rbf_feature = scatter_add(mul_output, batch, dim=0)
        output = (self.attention(rbf_feature)).squeeze(1)
        output = self.norm(self.ReLU(self.x2(output)))
        output = self.x3(output)
        return output


class EnhancedRBFModel(nn.Module):
    def __init__(self, hidden_size, num_kernels, n_filters, output_dim, dropratio, bias):
        super(EnhancedRBFModel, self).__init__()
        self.n_filters = n_filters
        self.RBFNN = RBFNN(hidden_size, num_kernels)

        self.RBFNNBlock = nn.ModuleList(
            [RBFNN(hidden_size, num_kernels) for _ in range(self.n_filters)])
        self.dropout = nn.Dropout(dropratio)
        self.mlp = nn.Sequential(
            nn.Linear(n_filters, output_dim, bias=bias),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropratio),
            nn.Linear(output_dim, output_dim, bias=bias),
            nn.BatchNorm1d(output_dim))

        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.RBFNNBlock]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self, x, pos, batch):
        global_out0 = []
        for i in range(self.n_filters):
            kernel_output = self.RBFNNBlock[i](x, pos)
            global_out0.append(kernel_output)

        global_out = torch.cat(global_out0, dim=1)

        # Global Pooling
        pooled_output = scatter_add(global_out, batch, dim=0)
        output = self.mlp(pooled_output)

        return output


class RBFAModel(nn.Module):
    def __init__(self, hidden_size, num_kernels, n_filters, output_dim, dropratio, bias):
        super(RBFAModel, self).__init__()
        self.n_filters = n_filters
        attention_size = hidden_size * 2

        self.RBFNN = RBFANN(hidden_size, attention_size, num_kernels)
        self.RBFNNBlock = nn.ModuleList(
            [RBFANN(hidden_size, attention_size, num_kernels) for _ in range(self.n_filters)])
        self.dropout = nn.Dropout(dropratio)
        self.mlp = nn.Sequential(
            nn.Linear(n_filters, output_dim, bias=bias),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropratio),
            nn.Linear(output_dim, output_dim, bias=bias),
            nn.BatchNorm1d(output_dim))

        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.RBFNNBlock]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self, x, pos, batch):
        global_out0 = []
        for i in range(self.n_filters):
            kernel_output = self.RBFNNBlock[i](x, pos, batch)
            global_out0.append(kernel_output)

        global_out = torch.cat(global_out0, dim=1)
        output = self.mlp(global_out)

        return output


class MetaEncoder(nn.Module):
    def __init__(self, x_ind, x_hs, edge_ind, e_hs, u_ind, u_hs, pos_ind, p_hs, dropratio, bias):
        super(MetaEncoder, self).__init__()
        self.ENPG = ENPGModel(x_ind, x_hs, edge_ind, e_hs, u_ind, u_hs, pos_ind, p_hs, dropratio, bias)

        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.ENPG]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self, x, edge_index, pos, edge_attr, u=None, batch=None, center=None):
        x, edge_attr, u, node_posemb = self.ENPG(x, edge_index, edge_attr, pos, pos, u, batch)
        edge_posemb = edge_attr
        return x, edge_attr, u, node_posemb, edge_posemb


class MetaGNN(nn.Module):
    def __init__(self, gnn_steps, x_ind, edge_ind, u_ind, pos_ind, dropratio, bias=True):
        super(MetaGNN, self).__init__()
        self.gnn_steps = gnn_steps
        self.attention = AttentionModule(u_ind * gnn_steps)
        self.ENPG = ENPGModel(x_ind, x_ind, edge_ind, edge_ind, u_ind, u_ind, pos_ind, pos_ind, dropratio, bias)

        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.attention, self.ENPG]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self, x, edge_index, pos, edge_attr, node_posemb, u, batch, center):
        global_out = []
        x0 = x
        edge_attr0 = edge_attr
        for i in range(self.gnn_steps):
            x1, edge_attr1, u1, node_posemb1 = self.ENPG(x, edge_index, edge_attr, pos, node_posemb, u, batch)
            node_posemb = node_posemb1 + node_posemb
            x = x1 + x
            edge_attr = edge_attr1 + edge_attr
            u = u1 + u
            global_out.append(u)
        global_out = torch.cat(global_out, dim=1)
        global_out = self.attention(global_out)
        return global_out


class GraphRBF(nn.Module):
    def __init__(self, gnn_steps, x_ind, edge_ind, x_hs, e_hs, u_hs, dropratio, bias, r_list, dist, max_nn):
        super(GraphRBF, self).__init__()
        self.dist = dist
        self.max_nn = max_nn
        self.u_ind = 0
        self.pos_ind = 3
        self.p_hs = 64
        self.bn = nn.ModuleList([nn.BatchNorm1d(x_ind),
                                 nn.BatchNorm1d(2)])
        self.encoder = MetaEncoder(x_ind=x_ind, x_hs=x_hs, edge_ind=edge_ind, e_hs=e_hs, u_ind=self.u_ind, u_hs=u_hs,
                                   pos_ind=self.pos_ind, p_hs=self.p_hs, dropratio=dropratio, bias=bias)
        self.meta_GN_gnn = MetaGNN(gnn_steps=gnn_steps, x_ind=x_hs, edge_ind=e_hs, u_ind=u_hs, pos_ind=self.p_hs,
                                   dropratio=dropratio, bias=True)
        self.clf = nn.Sequential(
            nn.Linear(u_hs * gnn_steps, u_hs // 2),
            nn.BatchNorm1d(u_hs // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropratio),
            nn.Linear(u_hs // 2, 2))

        self.pdist = nn.PairwiseDistance(p=2, keepdim=True)
        self.cossim = nn.CosineSimilarity(dim=1)

        self.r_list = r_list

    def forward(self, data):

        x, pos, batch = data.x, data.pos, data.batch
        pos_global, pos_side, pos_trans, res_psepos = data.pos_global, data.pos_side, data.pos_trans, data.res_psepos

        pos1 = pos.cpu()
        batch1 = batch.cpu()
        radius_index_list = radius_graph(pos1, r=self.r_list[0], batch=batch1, loop=True,
                                         max_num_neighbors=self.max_nn)
        center_index = []
        radius_index_list = radius_index_list.to(x.device)
        radius_attr_list = self.cal_edge_attr(radius_index_list, pos)

        center_distance = (torch.sqrt(torch.sum(res_psepos * res_psepos))) ** 2
        distance = torch.sqrt(torch.sum(pos * pos, dim=1)).unsqueeze(-1) / self.dist
        for i in range(x.size(0)):
            if distance[i] == 0:
                center_index.append(i)

        x = torch.cat([x, distance], dim=-1)
        x = self.bn[0](x.permute(1, 0).unsqueeze(0)).squeeze().permute(1, 0)
        pos0 = pos / self.r_list[0]
        pos1 = pos * center_distance

        x, radius_attr_list, u, node_posemb, edge_posemb = self.encoder(x=x, edge_index=radius_index_list,
                                                                        pos=pos_trans,
                                                                        edge_attr=radius_attr_list, u=None,
                                                                        batch=batch, center=center_index)

        global_output = self.meta_GN_gnn(x=x, edge_index=radius_index_list, pos=pos_trans, edge_attr=radius_attr_list,
                                         node_posemb=node_posemb, u=u, batch=batch, center=center_index)
        global_output = global_output.reshape(max(batch) + 1, -1)
        out = self.clf(global_output)
        # out = self.clf(u)
        out = F.softmax(out, -1)
        return out[:, 1]

    def cal_edge_attr(self, index_list, pos):
        radius_attr_list = torch.cat([self.pdist(pos[index_list[0]], pos[index_list[1]]) / self.r_list[0],
                                      (self.cossim(pos[index_list[0]], pos[index_list[1]]).unsqueeze(-1) + 1) / 2],
                                     dim=1)
        radius_attr_list = self.bn[1](radius_attr_list.permute(1, 0).unsqueeze(0)).squeeze().permute(1, 0)

        return radius_attr_list
