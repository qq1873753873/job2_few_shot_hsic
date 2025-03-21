import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class GraphSelfAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, adj, dropout, alpha, concat=True):
        super(GraphSelfAttentionLayer, self).__init__()
        self.dropout = dropout
        self.alpha = alpha

        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)), requires_grad=True)
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.to_linear = nn.Linear(out_features, out_features)

        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)), requires_grad=True)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.to_q = nn.Linear(out_features, out_features*1)
        self.to_kv = nn.Linear(out_features, out_features*2)
        self.to_qkv = nn.Linear(out_features, out_features*3)

        self.scale = out_features ** -0.5  # 1/sqrt(dim)
        self.heads = 4
        self.adj_bn = nn.BatchNorm2d(in_features)

    def forward(self, h, adj):#只需要特征矩阵和邻接矩阵
        # h.shape torch.Size([196, 128])  # 每个超像素包含一个长度为128的序列信息
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        Wh = self.to_linear(Wh)
        qkv = self.to_qkv(Wh).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'n (h d) -> h n d', h=self.heads), qkv)  # split into multi head attentions
        dots = torch.einsum('h i d, h j d -> h i j', q, k) * self.scale

        dots = F.softmax(dots, dim=-1)
        Wh0 = torch.einsum('h i j, h j d -> h i d', dots, v)
        Wh0 = rearrange(Wh0, 'h i d -> i (h d)')
        Wh0 = F.softmax(Wh0, dim=-1)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))  # 196 196 60 * 60 1  196 196 1
        zero_vec = -9e15 * torch.ones_like(e)
        dots = torch.sum(dots, dim=0)
        attention = torch.where(adj > 0, e + dots, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh) + Wh0

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]  # number of nodes
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


# source from https://github.com/PetarV-/GAT
class GAT(nn.Module):
    def __init__(self, nfeat, nhid, adj, nout, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.adj = adj  # 邻接矩阵
        self.attentions = [GraphSelfAttentionLayer(nfeat, nhid, adj, dropout=dropout, alpha=alpha, concat=True) for _ in
                            range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphSelfAttentionLayer(nhid * nheads, nout, adj, dropout=dropout, alpha=alpha, concat=False)
        self.norm = nn.LayerNorm(nhid * nheads)

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(torch.cat([att(x, self.adj) for att in self.attentions], dim=1))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, self.adj))
        return x


class CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()  # 1 128 145 145
        proj_query = x.view(m_batchsize, C, -1)  # 1 128 145*145
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)  # 1 145*145 128
        energy = torch.bmm(proj_query, proj_key)  # 1 128 128
        energy_new1 = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new1)
        proj_value = x.view(m_batchsize, C, -1)  # 1 128 145*145
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        return out


class SSConv(nn.Module):
    '''
    Spectral-Spatial Convolution
    '''
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(SSConv, self).__init__()
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )
        self.depth_conv = nn.Conv2d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=out_ch
        )
        self.Act1 = nn.LeakyReLU()
        self.Act2 = nn.LeakyReLU()
        self.BN = nn.BatchNorm2d(in_ch)

    def forward(self, input):
        out = self.BN(input)
        out = self.point_conv(out)
        out = self.Act1(out)

        out = self.depth_conv(out)
        out = self.Act2(out)
        return out


class SSConv1(nn.Module):
    '''
    Spectral-Spatial Convolution
    '''
    def __init__(self, in_ch, out_ch,kernel_size):
        super(SSConv1, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2,
            groups=out_ch,
            # dilation=2
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )
        self.Act1 = nn.LeakyReLU()
        self.Act2 = nn.LeakyReLU()
        self.BN = nn.BatchNorm2d(in_ch)

    def forward(self, input):
        out = self.point_conv(self.BN(input))
        out1 = self.Act1(out)
        out = self.depth_conv(out1)
        out = self.Act2(out)
        return out


class GS_GraphSAT(nn.Module):
    def __init__(self, height: int, width: int, changel: int, class_count: int, Q: torch.Tensor, A: torch.Tensor,
                 model='normal'):
        super(GS_GraphSAT, self).__init__()
        self.name = 'GS_GraphSAT'
        self.class_count = class_count  # 类别数
        self.channel = changel
        self.height = height
        self.width = width
        self.Q = Q
        self.A = A
        self.model = model
        self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True))  # 列归一化 Q
        self.WH = 0
        self.M = 2

        layers_count = 2

        self.CNN_denoise = nn.Sequential()
        for i in range(layers_count):
            if i == 0:
                self.CNN_denoise.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(self.channel))
                self.CNN_denoise.add_module('CNN_denoise_Conv' + str(i), nn.Conv2d(self.channel, 128, kernel_size=1))
                self.CNN_denoise.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())
            else:
                self.CNN_denoise.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(128), )
                self.CNN_denoise.add_module('CNN_denoise_Conv' + str(i), nn.Conv2d(128, 128, kernel_size=1))
                self.CNN_denoise.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())

        self.CNN_Branch = nn.Sequential()
        for i in range(layers_count):
            if i < layers_count - 1:
                self.CNN_Branch.add_module('Attention' + str(i), CAM_Module(128))
                self.CNN_Branch.add_module('CNN_Branch' + str(i), SSConv(128, 128, kernel_size=3))
                self.CNN_Branch.add_module('Drop_Branch'+str(i), nn.Dropout(0.2))
            else:
                self.CNN_Branch.add_module('Attention' + str(i), CAM_Module(128))
                self.CNN_Branch.add_module('CNN_Branch' + str(i), SSConv(128, 64, kernel_size=5))

        self.Attention = CAM_Module(128)

        self.CNN_Branch11 = SSConv(128, 128, kernel_size=1)
        self.CNN_Branch12 = SSConv(128, 64, kernel_size=3)
        self.CNN_Branch21 = SSConv1(128, 128, kernel_size=5)
        self.CNN_Branch22 = SSConv1(128, 64, kernel_size=7)

        self.GAT_Branch = nn.Sequential()
        self.GAT_Branch.add_module('GAT_Branch' + str(i),
                                    GAT(nfeat=128, nhid=32, adj=A, nout=64, dropout=0.4, nheads=4, alpha=0.2))

        self.linear1 = nn.Linear(64, 64)
        torch.nn.init.kaiming_normal_(self.linear1.weight, mode='fan_in', nonlinearity='leaky_relu')
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(64)

        self.Softmax_linear = nn.Sequential(nn.Linear(64, self.class_count))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 降维 - 升维：减少计算量
        self.fc = nn.Sequential(
            nn.Linear(128, 128 // 8, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(128 // 8, 128, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        (h, w, c) = x.shape  # 145 145 200
        # 降维 145 145 128
        noise = self.CNN_denoise(torch.unsqueeze(x.permute([2, 0, 1]), 0))  # 输入 1 200 145 145 输出 1 128 145 145
        noise = torch.squeeze(noise, 0).permute([1, 2, 0])
        clean_x = noise  # (145, 145, 128)
        # 构造超像素与像素之间的映射
        clean_x_flatten = clean_x.reshape([h * w, -1])  # 145*145 128
        superpixels_flatten = torch.mm(self.norm_col_Q.t(), clean_x_flatten)  # (196, 128) =(196, 21025) * (21025, 128)
        hx = clean_x#145,145,128

        # MAF
        hx = self.Attention(torch.unsqueeze(hx.permute([2, 0, 1]), 0))#([1, 128, 145, 145])
        CNN_result11 = self.CNN_Branch11(hx)#([1, 128, 145, 145])
        CNN_result21 = self.CNN_Branch21(hx)#([1, 128, 145, 145])
        CNN_result = CNN_result11 + CNN_result21# ([1, 128, 145, 145])
        CNN_result = self.Attention(CNN_result)

        CNN_result11_pool = self.avg_pool(CNN_result11).view(1, 128)
        CNN_result11_excite = CNN_result11 * self.fc(CNN_result11_pool).view(1, 128, 1, 1)
        CNN_result11 = CNN_result11 * CNN_result11_excite

        CNN_result21_pool = self.avg_pool(CNN_result21).view(1, 128)
        CNN_result21_excite = CNN_result21 * self.fc(CNN_result21_pool).view(1, 128, 1, 1)
        CNN_result21 = CNN_result21 * CNN_result21_excite

        CNN_result1 = CNN_result + CNN_result11
        CNN_result2 = CNN_result + CNN_result21

        CNN_result12 = self.CNN_Branch12(CNN_result1)
        CNN_result22 = self.CNN_Branch22(CNN_result2)
        CNN_result = CNN_result12 + CNN_result22
        CNN_result = self.Attention(CNN_result)

        CNN_result = CNN_result + CNN_result12 + CNN_result22 #都是 torch.Size([1, 64, 145, 145])
        CNN_result = torch.squeeze(CNN_result, 0).permute([1, 2, 0]).reshape([h * w, -1])# torch.Size([21025, 64])

        # 图注意力
        H = superpixels_flatten
        H = self.GAT_Branch(H)  # 输出 196 64
        GAT_result = torch.matmul(self.Q, H)  # (21025, 196) * (196, 64) = (21025, 64)
        GAT_result = self.linear1(GAT_result)
        GAT_result = self.act1(self.bn1(GAT_result))# 也是torch.Size([21025, 64])

        Y = 0.05 * CNN_result + 0.95 * GAT_result
        Y = self.Softmax_linear(Y)
        Y = F.softmax(Y, -1)
        return Y
