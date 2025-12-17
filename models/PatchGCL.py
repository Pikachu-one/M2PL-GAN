from __future__ import print_function
from packaging import version

import torch
from torch import nn
import math
import torch.nn.functional as F
from dgl import DGLGraph
from scipy import sparse
from dgl.nn.pytorch.factory import KNNGraph
from dgl.nn.pytorch import TAGConv, GATConv, GATv2Conv
import torch.nn.functional as F
import dgl.backend as B
import dgl.function as fn
import dgl
import numpy as np
from torch.nn import init
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import GATConv


eps = 1e-7

############
# utils
############

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net

############
# Functions
############

def cos_distance_softmax(x):
    soft = F.softmax(x, dim=2)
    w = soft.norm(p=2, dim=2, keepdim=True)
    return 1 - soft @ B.swapaxes(soft, -1, -2) / (w @ B.swapaxes(w, -1, -2)).clamp(min=eps)


def nonzero_graph(x, th, exist_adj=None):
    '''
    x: 节点特征，形状应为 [B, N, C]，即批次、节点数、特征维度。
    th: 阈值，控制边的存在与否（低于阈值的相似度将被舍弃）。
    exist_adj: 如果提供，就直接使用已存在的邻接矩阵（常用于 pooling 后继续用旧图结构）。
    '''
    if exist_adj is None:
        if B.ndim(x) == 2:
            x = B.unsqueeze(x, 0)
        
        n_samples, n_points, _ = B.shape(x)

        dist = torch.bmm(x, x.transpose(2,1)).squeeze().detach()
        base = torch.zeros_like(dist).cuda()
        base[dist>th] = 1
        adj = sparse.csr_matrix(B.asnumpy((base).squeeze(0)))
    else:
        adj = exist_adj
    g = DGLGraph(adj, readonly=True)

    return g, adj


class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        # print(x.shape)
        x = self.linear(x)
        x = self.l2norm(x)
        return x

class NCESoftmaxLoss(nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""
    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss().cuda()

    def forward(self, x):
        bsz = x.shape[0]
        
        label = torch.zeros([bsz]).cuda().long()
        loss = self.criterion(x, label)
        return loss


class Normalize(nn.Module):
    """normalization layer"""

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


def top_k_graph(scores, g, h, k):
    '''
    :param scores: Node score
    :param g: teacher adjacent matrix
    :param h: Node feature
    :param k: Number of pooled node
    :return: pooled Adj, p
    '''
    ## get high scored (score, node) = (values, new_h)
    values, idx = torch.topk(scores, max(2, int(k)))
    new_h = h[idx, :]  ## (k, dim)
    values = torch.unsqueeze(values, -1)  ## (k, score)
    new_h = torch.mul(new_h, values)  ## (k, dim)

    ## get pooled adjacent matrix g
    ## increase connectiviy by hop2 as in original papaer.
    un_g = g.bool().float()
    un_g = torch.matmul(un_g, un_g).bool().float()
    un_g = un_g[idx, :]  ## select idx rows
    un_g = un_g[:, idx]  ## select idx column
    g = norm_g(un_g)  ## random work by 1-hot graph
    g = sparse.csr_matrix(g)  ## g is weighted graph
    return g, new_h, idx

def norm_g(g):
    degrees = torch.sum(g, 1)
    g = g / degrees
    return g

###########
# Loss
###########
class GNNLoss(nn.Module):
    def __init__(self, opt, use_mlp=True):
        super(GNNLoss, self).__init__()

        ## graph arguments
        self.num_hop = 4
        self.pooling_num = 0
        self.down_scale = 8
        self.pooling_ratio = '1, 1, 1'
        self.nonzero_th = 0.6
        
        self.gpu_ids = opt.gpu_ids
        self.nc = opt.netF_nc
        self.num_patch = opt.num_patches
        self.init_type = 'normal'
        self.init_gain = 0.02

        self.use_mlp = use_mlp
        self.mlp_init = False
        
        self.conv_type = opt.conv_type
        
        self.criterion = NCESoftmaxLoss()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')

    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            embed = Embed(input_nc, self.nc)
            '''
            pools = nn.ModuleList()
            #gnn_pools = nn.ModuleList()
            if self.pooling_num>0:
                for i in range(self.pooling_num):
                    pools.append(Pool(self.nc))
                    #gnn_pools.append(Encoder(self.nc, self.nc,1))
            '''
            gnn = Encoder(self.nc, self.nc, self.num_hop,conv_type=self.conv_type)
            
            if len(self.gpu_ids) > 0:
                gnn.cuda()
                #pools.cuda()
                #gnn_pools.cuda()
                #embed.cuda()
                
            setattr(self, 'gnn_%d' % mlp_id, gnn)
            setattr(self, 'embed_%d' % mlp_id, embed)
            #setattr(self, 'pools_%d' % mlp_id, pools)
            #setattr(self, 'gnn_pools_%d' % mlp_id, gnn_pools)

        init_net(self, self.init_type, self.init_gain, self.gpu_ids)
        self.mlp_init = True
        
    def calc_gnn(self, f_es, f_et, num_patches, gnn=None, adj_s=None, adj_t=None):
        batch_dim_for_bmm=1
        T = 0.07

        ## input features
        # 基于特征之间的相似性构建图结构。
        # self.nonzero_th：构图的阈值（比如余弦相似度 > 0.7 才建边）。
        # exist_adj：可复用已有邻接矩阵（加速）。
        G_pos_t,adj_t = nonzero_graph(f_et.detach(), self.nonzero_th, exist_adj=adj_t)  
        G_pos_t = G_pos_t.to(f_es.device)
        G_pos_t = dgl.add_self_loop(G_pos_t)                                 # 给图中每个节点添加自环，确保节点自己的信息也参与卷积。
        G_pos_t.ndata['h'] = f_et                                            # 把节点的原始特征 f_et 存到图的节点数据中，键是 'h'（很多 GNN 模型默认使用 'h'）。
        f_gt = gnn(G_pos_t)                                                  # 将图送入 GNN 网络，提取每个节点的特征表示。
        f_gt = f_gt.detach()

        ## output features
        G_pos_s,adj_s = nonzero_graph(f_es.detach(), self.nonzero_th, exist_adj=adj_t)
        G_pos_s = G_pos_s.to(f_es.device)
        G_pos_s = dgl.add_self_loop(G_pos_s)
        G_pos_s.ndata['h'] = f_es
        f_gs = gnn(G_pos_s)                                                  ##  将图送入 GNN 网络，提取每个节点的特征表示。

        ## node-wise contrastive loss
        f_gt = f_gt.squeeze()                                                # target图上每个节点的GNN输出，shape: [N, C]
        f_gs = f_gs.squeeze()                                                # source图上每个节点的GNN输出，shape: [N, C]
        
        #计算负样本相似度矩阵： bmm 是 batch 矩阵乘法，这里是所有节点对之间的相似度矩阵。
        f_gt_reshape = f_gt.view(batch_dim_for_bmm,-1,self.nc)               # [1, N, C]
        f_gs_reshape = f_gs.view(batch_dim_for_bmm,-1,self.nc)               # [1, N, C]
    
        diagonal = torch.eye(num_patches, device=f_es.device, dtype=torch.bool)[None, :, :] 
        
        gs_pos = torch.einsum('nc,nc->n', [f_gt, f_gs]).unsqueeze(-1)        # einsum('nc,nc->n')：对同一位置的节点做点积，计算正样本对相似度（类似余弦相似，但未归一化）。
        gs_neg = torch.bmm(f_gt_reshape, f_gs_reshape.transpose(2, 1))       # [1, N, N]  gs_neg[i][j] 表示 f_gt[i] ⋅ f_gs[j]，是 i 和 j 节点之间的相似度
        gs_neg.masked_fill_(diagonal, -10.0)                                 # 用 -10.0 避免正样本误作为负样本参与 softmax（后面会被 exponentiation 降权）。
        gs_neg = gs_neg.view(-1, num_patches) 
        out_gs = torch.cat([gs_pos, gs_neg], dim=1)                          # [N, 1+N]
        out_gs = torch.div(out_gs, T)
        out_gs = out_gs.contiguous()
        loss_gs = self.criterion(out_gs)  #✅ 两个方向都考虑，有助于双向特征对齐（bi-directional contrastive learning），更稳定、更全面。
        '''
        '''
        gt_pos = torch.einsum('nc,nc->n', [f_gs, f_gt]).unsqueeze(-1)
        gt_neg = torch.bmm(f_gs_reshape, f_gt_reshape.transpose(2, 1))       # [1, N, N]
        gt_neg.masked_fill_(diagonal, -10.0)
        gt_neg = gt_neg.view(-1, num_patches)  
        out_gt = torch.cat([gt_pos, gt_neg], dim=1)
        out_gt = torch.div(out_gt, T)
        out_gt = out_gt.contiguous()
        loss_gt = self.criterion(out_gt)  #✅ 两个方向都考虑，有助于双向特征对齐（bi-directional contrastive learning），更稳定、更全面。
        

        #loss_g = loss_gs 
        #loss_g = loss_gt 
        loss_g = loss_gs + loss_gt

        return loss_g, f_gs, f_gt, adj_s, adj_t

    def forward(self, feat_s, feat_t, num_patches=64):

        loss_g_total = 0
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feat_s)
        '''
        for i, feat in enumerate(feat_s):
            print(f"Layer {i}: shape = {feat.shape}")

        for i, feat in enumerate(feat_t):
            print(f"Layer {i}: shape = {feat.shape}")   
        '''    
        for mlp_id, (f_es_embed, f_et_embed) in enumerate(zip(feat_s, feat_t)):

            loss_g = 0
            
            gnn = getattr(self, 'gnn_%d' % mlp_id)
            #pools = getattr(self, 'pools_%d' % mlp_id)
            #gnn_pools = getattr(self, 'gnn_pools_%d' % mlp_id)
            
            '''
            '''
            embed = getattr(self, 'embed_%d' % mlp_id)
            num_patches=f_es_embed.shape[0]
            f_et_embed = embed(f_et_embed)
            f_es_embed = embed(f_es_embed)
            
            ### graph loss before pooling
            loss_gnn, gnn_s, gnn_t, adj_s, adj_t = self.calc_gnn(f_es_embed, f_et_embed, num_patches, gnn=gnn)
            ### pooling
            f_es = gnn_s
            f_et = gnn_t
            
            loss_g_total += loss_gnn
        loss_g_total /= len(feat_s)

        return loss_g_total



class Encoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_hop, conv_type='tag'):
        super(Encoder, self).__init__()
        self.conv_type = conv_type
        #Topology Adaptive Graph Convolution 图神经网络（GNN）中的编码器模块, 利用节点的多个不同阶数（0-hop, 1-hop, ..., K-hop）邻居的信息，自动加权聚合
        if self.conv_type == 'tag':
            self.conv1 = TAGConv(in_dim, hidden_dim, k=num_hop)
        elif self.conv_type == 'gcn':
            self.conv1 = GraphConv(in_dim, hidden_dim)
        elif self.conv_type == 'gat':    
            self.conv1 = GATConv(in_dim, hidden_dim // 4, num_heads=4)
        elif self.conv_type == 'none':
            self.conv1 = None     # 不使用任何图卷积

        self.l2norm = Normalize(2)

    def forward(self, g, edge_weight=None):

        h = g.ndata['h'] # 读取每个节点的初始特征 h。 

        if self.conv_type == 'tag':
            h = self.conv1(g, h, edge_weight)

        elif self.conv_type == 'gcn':
            h = self.conv1(g, h)     # 不能传 edge_weight

        elif self.conv_type == 'gat':
            h = self.conv1(g, h)     # 输出: [N, heads, out_dim]
            h = h.flatten(1)          
            
        elif self.conv_type == 'none':
            pass    
        
        h = self.l2norm(h)
        return h


class Pool(nn.Module):
    def __init__(self,  in_dim):
        super(Pool, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        
    def forward(self, ht, hs, g, k,):
        '''
        :param ht: teacher feature
        :param hs: student feature
        :param g: teacher adjacent matrix
        :param k: Number of pooled node
        :return: pooled teacher feat, pooled student feat, pooled adjacency matrix
        '''
        g_coo = g.tocoo()
        g_data = torch.sparse.LongTensor(torch.LongTensor([g_coo.row.tolist(), g_coo.col.tolist()]), torch.LongTensor(g_coo.data.astype(np.int32))).to_dense()

        ## scores for each node
        weights = self.proj(ht).squeeze()
        scores = self.sigmoid(weights)

        ## pool graphs by scores
        g, new_ht, idx = top_k_graph(scores, g_data, ht, k)
        _, new_hs, _ = top_k_graph(scores, g_data, hs, k)
        return new_ht, new_hs, g


