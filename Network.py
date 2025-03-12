import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from models.BFAM import BFAM
from models.MSAA import MSAA

def drop_edge_dense(adj, drop_rate=0.2):
    """
    随机丢弃图中的一部分边（稠密图版本）。
    
    参数:
        adj (torch.Tensor): 图的邻接矩阵，形状为 (num_nodes, num_nodes)。
        drop_rate (float): 丢弃边的比例，默认为 0.2。
    
    返回:
        new_adj (torch.Tensor): 修改后的邻接矩阵。
    """
    num_nodes = adj.size(0)  # 节点数量
    num_edges = adj.nonzero().size(0)  # 边的数量
    num_drop = int(num_edges * drop_rate)  # 需要丢弃的边数量

    # 获取所有边的索引
    edge_indices = adj.nonzero(as_tuple=False)

    # 随机选择要丢弃的边
    drop_indices = torch.randperm(num_edges)[:num_drop]
    drop_edges = edge_indices[drop_indices]

    # 将丢弃的边对应的邻接矩阵值设为 0
    new_adj = adj.clone()
    new_adj[drop_edges[:, 0], drop_edges[:, 1]] = 0

    return new_adj

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

    def forward(self, h, adj):
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
    
class CNNConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, k, h, w):
        super(CNNConvBlock, self).__init__()
        self.BN = nn.BatchNorm2d(ch_in)
        self.conv_in = nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, stride=1, groups=1)
        self.conv_out = nn.Conv2d(ch_out, ch_out, kernel_size=k, padding=k//2, stride=1, groups=ch_out)#64 64 7 3 1 64
        self.pool = nn.AvgPool2d(3, padding=1, stride=1)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        #print("conv start\n")
        x = self.BN(x)#145*145*200
        #print(x.shape)
        x = self.act(self.conv_in(x))#145*145*64
        #print(x.shape)
        x = self.pool(x)#145*145*64
        #print(x.shape)
        x = self.act(self.conv_out(x))#还是145*145*64
        #print(x.shape)
        #print("end\n")
        return x

class GS_GraphSAT(nn.Module):
    def __init__(self, height: int, width: int, channel: int, class_count: int, Q: torch.Tensor, A: torch.Tensor,
                 model='normal', CNN_nhid=64):
        super(GS_GraphSAT, self).__init__()
        self.name = 'GS_GraphSAT'
        self.class_count = class_count  # 类别数
        self.channel = channel
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

        # 多尺度CNN
        # self.CNNlayerA1 = CNNConvBlock(128, CNN_nhid, 3, self.height, self.width)#128->64
        # self.CNNlayerA2 = CNNConvBlock(CNN_nhid, CNN_nhid, 3, self.height, self.width)
        # self.CNNlayerA3 = CNNConvBlock(CNN_nhid, CNN_nhid, 3, self.height, self.width)
        # self.CNNlayerB1 = CNNConvBlock(128, CNN_nhid, 5, self.height, self.width)#input 200 64 5 145 145
        # self.CNNlayerB2 = CNNConvBlock(CNN_nhid, CNN_nhid, 5, self.height, self.width)
        # self.CNNlayerB3 = CNNConvBlock(CNN_nhid, CNN_nhid, 5, self.height, self.width)
        # self.CNNlayerC1 = CNNConvBlock(128, CNN_nhid, 7, self.height, self.width)#input 200 64 3 145 145
        # self.CNNlayerC2 = CNNConvBlock(CNN_nhid, CNN_nhid, 7, self.height, self.width)
        # self.CNNlayerC3 = CNNConvBlock(CNN_nhid, CNN_nhid, 7, self.height, self.width)
        # self.MSAA=MSAA(64, 64)

        # 原CNN
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
        #BFAM
        self.bfam = BFAM(inp=128, out=64)
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
        hx = clean_x

        # #多尺度CNN
        # CNNin = torch.unsqueeze(hx.permute([2, 0, 1]), 0)  # 1 128 145 145
        # CNNmid1_A = self.CNNlayerA1(CNNin) 
        # CNNmid1_B = self.CNNlayerB1(CNNin)
        # CNNmid1_C = self.CNNlayerC1(CNNin)
        # CNNin = CNNmid1_A + CNNmid1_B + CNNmid1_C
        # # CNNmid2_A = self.CNNlayerA2(CNNin)
        # # CNNmid2_B = self.CNNlayerB2(CNNin)
        # # CNNmid2_C = self.CNNlayerC2(CNNin)
        # # CNNin = CNNmid2_A + CNNmid2_B + CNNmid2_C
        # CNNout_A = self.CNNlayerA3(CNNin)
        # CNNout_B = self.CNNlayerB3(CNNin)
        # CNNout_C = self.CNNlayerC3(CNNin)
        # #三次卷积块出来之后的数据都是1*64*145*145形式的，输入MSAA模块
        # CNN_result=CNNout_A+CNNout_B+CNNout_C
        # #CNN_result=self.MSAA(CNNout_A,CNNout_B,CNNout_C)
        

        # 原注意力
        hx = self.Attention(torch.unsqueeze(hx.permute([2, 0, 1]), 0)) # MAF
        CNN_result11 = self.CNN_Branch11(hx)
        CNN_result21 = self.CNN_Branch21(hx)
        CNN_result = CNN_result11 + CNN_result21
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
        CNN_result = CNN_result + CNN_result12 + CNN_result22#torch.Size([1, 64, 145, 145])
        CNN_HWC=torch.squeeze(CNN_result, 0).permute([1, 2, 0])
        CNN_BCHW=CNN_result
        CNN_result = CNN_HWC.reshape([h * w, -1])

        # 图注意力
        H = superpixels_flatten
        H = self.GAT_Branch(H)  # 输出 196 64
        GAT_result = torch.matmul(self.Q, H)  # (21025, 196) * (196, 64) = (21025, 64)
        GAT_HWC=GAT_result.reshape([h, w, -1])
        GAT_BCHW=GAT_HWC.permute([2, 0, 1]).unsqueeze(0)
        GAT_result = self.linear1(GAT_result)
        GAT_result = self.act1(self.bn1(GAT_result))
        #使用BFAM模块来融合
        # Y=self.bfam(CNN_BCHW, GAT_BCHW) #1,64,145,145
        # Y=Y.reshape([h * w, -1])
        #自适应融合
        Y = 0.05 * CNN_result + 0.95 * GAT_result#torch.Size([21025, 64])
        #融合CNN_result和GAT_result
        Y = self.Softmax_linear(Y)#torch.Size([21025, 16])
        Y = F.softmax(Y, -1)#对Y的最后一个维度应用了softmax函数。变成21025, 1
        return Y
