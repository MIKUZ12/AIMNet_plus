import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import math


class MLP(nn.Module):
    # 100/1000..., 512, []
    def __init__(self, in_features, out_features, hidden_features=[], batchNorm=False,
                 nonlinearity='leaky_relu', negative_slope=0.1,
                 with_output_nonlineartity=True):
        super(MLP, self).__init__()
        self.nonlinearity = nonlinearity
        self.negative_slope = negative_slope
        # 创建容器
        self.fcs = nn.ModuleList()
        if hidden_features:
            in_dims = [in_features] + hidden_features
            out_dims = hidden_features + [out_features]
            for i in range(len(in_dims)):
                self.fcs.append(nn.Linear(in_dims[i], out_dims[i]))
                if with_output_nonlineartity or i < len(hidden_features):
                    if batchNorm:
                        self.fcs.append(nn.BatchNorm1d(out_dims[i], track_running_stats=True))
                    if nonlinearity == 'relu':
                        self.fcs.append(nn.ReLU(inplace=True))
                    elif nonlinearity == 'leaky_relu':
                        self.fcs.append(nn.LeakyReLU(negative_slope, inplace=True))
                    else:
                        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))
        # 上面没有用，只用了这个
        else:
            self.fcs.append(nn.Linear(in_features, out_features))
            if with_output_nonlineartity:
                # 一直都是false
                if batchNorm:
                    self.fcs.append(nn.BatchNorm1d(out_features, track_running_stats=True))
                if nonlinearity == 'relu':
                    self.fcs.append(nn.ReLU(inplace=True))
                elif nonlinearity == 'leaky_relu':
                    self.fcs.append(nn.LeakyReLU(negative_slope, inplace=True))
                else:
                    raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))

        self.reset_parameters()

    def reset_parameters(self):
        for l in self.fcs:
            if l.__class__.__name__ == 'Linear':
                nn.init.kaiming_uniform_(l.weight, a=self.negative_slope,
                                         nonlinearity=self.nonlinearity)
                if self.nonlinearity == 'leaky_relu' or self.nonlinearity == 'relu':
                    nn.init.uniform_(l.bias, 0, 0.1)
                else:
                    nn.init.constant_(l.bias, 0.0)
            elif l.__class__.__name__ == 'BatchNorm1d':
                l.reset_parameters()

    def forward(self, input):
        for l in self.fcs:
            input = l(input)
        return input


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        ## e是输出的注意力分数 e: (N, N) 表示节点j对节点i的重要性
        e = self._prepare_attentional_mechanism_input(Wh)
        ## 近似一个很小的值，仅保留有边连接的注意力分数，其余位置赋值为zero-vec的值
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        ## 对每一行做softmax归一化，每一行都代表数据的概率分布 (N, N),含义是其他节点对当前节点的权值之和为1
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        ## 计算注意力分数（a[Whi||Whj]）
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.2, alpha=0.2, nheads=8):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        # 这里就考虑了多头的自注意，attentins是一个有多头注意力层的列表
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        # 注意此处的构造，经过上述多头注意力的输出，形状是：nheads * nhid
        self.out_att = GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        # 多头注意力，根据输入的embedding和adj计算其注意力分数α矩阵（对应公式2）
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        # 先将计算得到的α再进行一次注意力的计算，然后经过激活函数输出h（对应公式3）
        x = F.elu(self.out_att(x, adj))
        # return F.log_softmax(x, dim=1)
        # x: nhid
        return x


class GIN(nn.Module):
    def __init__(self, num_layers, in_features, out_features, hidden_features=[],
                 eps=0.0, train_eps=True, residual=True, batchNorm=True,
                 nonlinearity='leaky_relu', negative_slope=0.1):
        super(GIN, self).__init__()

        self.GINLayers = nn.ModuleList()

        if in_features != out_features:
            first_layer_res = False
        else:
            first_layer_res = True
        self.GINLayers.append(GINLayer(MLP(in_features, out_features, hidden_features, batchNorm,
                                           nonlinearity, negative_slope),
                                       eps, train_eps, first_layer_res))
        for i in range(num_layers - 1):
            self.GINLayers.append(GINLayer(MLP(out_features, out_features, hidden_features, batchNorm,
                                               nonlinearity, negative_slope),
                                           eps, train_eps, residual))
        self.reset_parameters()

    def reset_parameters(self):
        for l in self.GINLayers:
            l.reset_parameters()

    def forward(self, input, adj):
        for l in self.GINLayers:
            input = l(input, adj)
        return input


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # scores shape is [bs heads view view]
    # print('scores',scores.shape)
    if mask is not None:
        mask = mask.float()
        mask = mask.unsqueeze(-1).matmul(mask.unsqueeze(-2))  # mask shape is [bs 1 view view]
        # mask = mask.unsqueeze(1) #mask shape is [bs 1 1 view]
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output


class FDModel(nn.Module):
    # in_features_x是特征数，是那个六维数组
    def __init__(self, d_list, in_features_y, hidden_features, out_features, beta=0.2,
                 in_layers1=1, out_layers=1, batchNorm=False,
                 nonlinearity='leaky_relu', negative_slope=0.1):
        super(FDModel, self).__init__()
        # hid_features = 512
        hidden_list = [hidden_features] * (in_layers1 - 1)

        # 这个hidden_features实际上是定义中的out_features, 所以是512, 上面的hidden_features和out_features都是512
        # in_features_x是特征数，是那个六维数组
        # 100/1000..., 512, []
        ## MLP的模型表，输入是dim_pre_view，输出是hidden_features
        self.MLP_list = nn.ModuleList([MLP(dim_pre_view, hidden_features, hidden_list,
                                           batchNorm, nonlinearity, negative_slope) for dim_pre_view in d_list])
        ## MLP的模型表，输入是hidden_features，输出是out_features
        self.MLP_list2 = nn.ModuleList([MLP(hidden_features, dim_pre_view, hidden_list,
                                            batchNorm, nonlinearity, negative_slope) for dim_pre_view in d_list])
        # in_features = class_emb
        self.NN2 = nn.Linear(in_features_y, hidden_features)
        # 超参
        self.beta = beta
        assert self.beta > 0
        hidden_list = [hidden_features] * (out_layers - 1)

        self.Q = nn.Linear(hidden_features, out_features)
        self.K = nn.Linear(hidden_features, out_features)
        self.V = nn.Linear(hidden_features, out_features)
        self.out = nn.Linear(hidden_features, out_features)
        self.norm = Norm(hidden_features)
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.MLP_list:
            module.reset_parameters()
        for module in self.MLP_list2:
            module.reset_parameters()
        nn.init.kaiming_uniform_(self.NN2.weight, nonlinearity='sigmoid')
        nn.init.constant_(self.NN2.bias, 0.0)

    # def forward(self, x, y, mask=None, mode = None):
    #     x_processed = []
    #     # 利用MLP提取多视图的嵌入特征 X -> Z
    #     # 输入的x是data[]，一个长为6（view的个数）列表，其中的元素是已经经过掩码处理过的输入数据（也是列表）
    #     for i in range(len(x)):
    #         x_i = self.MLP_list[i](x[i])
    #         x_processed.append(x_i)
    #     x_processed = torch.stack(x_processed, dim=0)  # m n d
    #     d = x_processed.shape[-1]
    #     # x_q是计算出来的多视图特征嵌入
    #     x_q = x_processed
    #     # 计算注意力分数（对应公式4）
    #     x_q = F.normalize(x_q, dim=-1)
    #     x_att_score = x_q.matmul(x_q.transpose(1, 2))  # m n n
    #     x_att_score = (x_att_score / self.beta).exp()
    #     # 处理缺失数据的mask
    #     if mask is not None:
    #         # 应用mask（对应公式5），这一步只用来推断，不需要梯度回传
    #         # 有的样本没有对应的view数据，因此其计算出来的x_att_score是缺失的， 此处利用mask以及最大池化，保留原有x_att_score的数据并提取全局最大的注意力值
    #         mask_12 = torch.matmul(mask.float().t().unsqueeze(dim=-1), mask.float().t().unsqueeze(dim=1))  # m n n
    #         mask_12 = mask_12.to(x_att_score.device)
    #         x_att_score = x_att_score.mul(mask_12)
    #         x_att_score = x_att_score.max(dim=0).values
    #         # 计算置信度（对应公式9），采取的是最大池化之后的x_att_score
    #         # 置信度是用来约束or训练补全的性能的，置信度高则补全特征和相邻特征高度相关，补全结果可信
    #         # 各个view的预测由置信度加权，训练补全质量变高
    #         x_att_score = x_att_score.fill_diagonal_(0.) # 对角元素置0，在置信度计算中不考虑自连接
    #         ## 在此已经对x_att_score做了归一化
    #         x_att_score = x_att_score / (x_att_score.sum(dim=-1, keepdim=True) + 1e-9)
    #         mask = mask.to(x_att_score.device)
    #         x_att_score = x_att_score.unsqueeze(0).mul(mask.t().unsqueeze(1))  #应用了掩码计算注意力分数 m n n
    #     # 基于注意力机制进行聚合（对应公式6），至此已经计算出来了对于输入x，各个视图的重构特征x_att_score
    #     new_x = x_att_score.matmul(x_processed)
    #     # 对原始的输入x中缺失view特征的部分进行动态补全（对应公式7）
    #     new_x = new_x * (1 - mask.transpose(0, 1).unsqueeze(dim=-1)) * 1 + x_processed * mask.transpose(0, 1).unsqueeze(
    #         dim=-1)
        
    #     # 结合标签嵌入特征（做一个线性映射）
    #     # 此处是对应公式8，对嵌入好的标签特征先经过一个线性层然后做sigmoid
    #     y_n = self.NN2(y).sigmoid_()  # b2 x h
    #     # Z = []
    #     # # 求出来交互特征（对应公式8）
    #     # for i in range(new_x.shape[0]):
    #     #     z_i = new_x[i].unsqueeze(1) * y_n.unsqueeze(0)
    #     #     Z.append(z_i)
    #     x_new_processed = []
    #     for i in range(len(x)):
    #         x_new_i = self.MLP_list2[i](new_x[i])
    #         x_new_processed.append(x_new_i)
    #     x_new_processed = [torch.nan_to_num(x, nan=0.0) for x in x_new_processed]

    #     if mode == 'train':
    #         # 创建输入数据的副本，避免修改原始数据
    #         masked_input = []
    #         prop = 0.15  # 掩码比例，可以根据需要调整
            
    #         for X in x_new_processed:
    #             # 计算要掩码的长度
    #             mask_len = int(prop * X.size(-1))
    #             # 为每个样本生成随机起始位置
    #             st = torch.randint(low=0, high=X.size(-1)-mask_len-1, size=(X.size(0),)).to(X.device)
    #             # 创建掩码张量
    #             random_mask = torch.ones_like(X).to(X.device)
    #             # 应用掩码
    #             for j in range(X.size(0)):
    #                 random_mask[j, st[j]:st[j]+mask_len] = 0
    #             # 应用掩码到数据
    #             masked_X = X * random_mask
    #             masked_input.append(masked_X)
            
    #         # 使用掩码后的数据
    #         x_new_processed = masked_input

    #     return  new_x, x_new_processed, y_n

    # def forward(self, x, y, mask=None, mode=None):
    #     device = x[0].device
    #     # 直接使用原始数据，不通过MLP进行特征提取
    #     x_original = x
    #     num_views = len(x_original)
    #     for i in range(num_views):
    #         x_original[i] = x_original[i].to(device)
    #     batch_size = x_original[0].size(0)
    #     if y.device != device:
    #         y = y.to(device)
    #     if mask is not None and mask.device != device:
    #         mask = mask.to(device)
    #     # 计算每个视图的样本间相似度
    #     view_sims = []
    #     for i in range(num_views):
    #         # 对当前视图内的特征进行归一化
    #         x_view_norm = F.normalize(x_original[i], p=2, dim=1)
            
    #         # 计算当前视图内的样本间相似度
    #         view_sim = torch.matmul(x_view_norm, x_view_norm.transpose(0, 1))  # [batch_size, batch_size]
    #         view_sims.append(view_sim)
        
    #     # 将所有视图的相似度堆叠成一个张量
    #     stacked_sims = torch.stack(view_sims, dim=0)  # [num_views, batch_size, batch_size]
        
    #     # 应用温度缩放
    #     x_att_score = torch.exp(stacked_sims / self.beta)
        
    #     # 处理缺失数据的mask
    #     if mask is not None:
    #         # 与原代码一致：创建mask矩阵并应用
    #         mask_12 = torch.matmul(mask.float().t().unsqueeze(dim=-1), mask.float().t().unsqueeze(dim=1))  # m n n
    #         mask_12 = mask_12.to(x_att_score.device)
    #         x_att_score = x_att_score.mul(mask_12)
            
    #         # 与原代码一致：在视图维度上进行最大池化
    #         pooled_att_score = x_att_score.max(dim=0)[0]  # [batch_size, batch_size]
            
    #         # 与原代码一致：对角线元素置0
    #         pooled_att_score = pooled_att_score.fill_diagonal_(0.)
            
    #         # 与原代码一致：归一化注意力分数
    #         pooled_att_score = pooled_att_score / (pooled_att_score.sum(dim=-1, keepdim=True) + 1e-9)
            
    #         # 与原代码一致：将注意力分数扩展为多个视图
    #         # 但不能用unsqueeze和mul，因为每个视图的维度可能不同
    #         # 转到同一个设备：
    #         pooled_att_score = pooled_att_score.to(device)
    #         pooled_att_score = pooled_att_score.unsqueeze(0).mul(mask.t().unsqueeze(1))
    #         # view_att_scores = []
    #         # for i in range(num_views):
    #         #     view_mask = mask[:, i].float().to(pooled_att_score.device)  # [batch_size]
    #         #     view_att = pooled_att_score * view_mask.unsqueeze(1)  # [batch_size, batch_size]
    #         #     view_att_scores.append(view_att)
    #     else:
    #         # 如果没有mask，直接使用注意力分数
    #         pooled_att_score = x_att_score.max(dim=0)[0]
    #         diag_mask = 1.0 - torch.eye(batch_size, device=pooled_att_score.device)
    #         pooled_att_score = pooled_att_score * diag_mask
    #         pooled_att_score = pooled_att_score / (pooled_att_score.sum(dim=-1, keepdim=True) + 1e-9)
    #         view_att_scores = [pooled_att_score.clone() for _ in range(num_views)]
        
    #     # 对每个视图单独计算和应用补全
    #     new_x = []
    #     for i in range(num_views):
    #         # 使用当前视图的注意力分数进行加权聚合
    #         # 转到同一设备：
    #         current_new_x = torch.matmul(pooled_att_score[i], x_original[i])  # [batch_size, feature_dim_i]
            
    #         # 对缺失部分进行动态补全
    #         if mask is not None:
    #             # 创建当前视图的掩码
    #             view_mask = mask[:, i].float().unsqueeze(1).to(current_new_x.device)  # [batch_size, 1]
    #             view_mask = view_mask.expand(-1, x_original[i].size(1))  # [batch_size, feature_dim_i]
                
    #             # 对原始输入中缺失view特征的部分进行动态补全
    #             current_new_x = current_new_x * (1 - view_mask) + x_original[i] * view_mask
            
    #         new_x.append(current_new_x)
        
    #     # 与原代码一致：结合标签嵌入特征
    #     y_n = self.NN2(y)
    #     y_n = torch.sigmoid(y_n)  # 使用非原地操作替代sigmoid_()
        
    #     # 处理x_new_processed，但不使用MLP
    #     x_new_processed = []
    #     for i in range(num_views):
    #         # 直接使用new_x，不通过MLP_list2
    #         x_new_i = new_x[i]
    #         # 处理NaN值
    #         x_new_i = torch.where(torch.isnan(x_new_i), torch.zeros_like(x_new_i), x_new_i)
    #         x_new_processed.append(x_new_i)
        
    #     # 训练模式下应用随机掩码
    #     if mode == 'train':
    #         masked_input = []
    #         prop = 0.15  # 掩码比例
            
    #         for X in x_new_processed:
    #             # 计算要掩码的长度
    #             mask_len = int(prop * X.size(-1))
    #             if mask_len < 1:
    #                 mask_len = 1
                    
    #             # 处理边界情况
    #             max_start = max(0, X.size(-1) - mask_len - 1)
    #             if max_start <= 0:
    #                 # 特征维度太小，无法应用随机掩码
    #                 masked_input.append(X)
    #                 continue
                    
    #             # 为每个样本生成随机起始位置
    #             st = torch.randint(low=0, high=max_start, size=(X.size(0),), device=X.device)
                
    #             # 创建掩码张量
    #             random_mask = torch.ones_like(X, device=X.device)
                
    #             # 应用掩码
    #             for j in range(X.size(0)):
    #                 end_idx = min(st[j] + mask_len, X.size(-1))
    #                 random_mask[j, st[j]:end_idx] = 0
                    
    #             # 应用掩码到数据
    #             masked_X = X * random_mask
    #             masked_input.append(masked_X)
            
    #         # 使用掩码后的数据
    #         x_new_processed = masked_input
        
    #     return new_x, x_new_processed, y_n
    def forward(self, x, y, mask=None, mode=None):
        top_k=5
        N = x[0].size(0)  # 每个视图的样本数
        view_num = len(x)
        x_att_score = []
        for v in range(view_num):
            x_q = F.normalize(x[v], dim=-1)  # 对每个视图的x进行归一化
            att_score = x_q.matmul(x_q.transpose(0, 1))  # 计算m n n形状的注意力分数
            att_score = (att_score / self.beta).exp()  # 缩放并计算指数
            x_att_score.append(att_score)
        # 将每个视图的注意力分数拼接成一个V × N × N的Tensor
        x_att_score=torch.stack(x_att_score, dim=0)
        confi = (torch.log(x_att_score + 1e-9) * self.beta)
        x_att = x_att_score.max(dim=0).values
        # 将所有设备统一
        x_att = x_att.to(x[0].device)
        confi = confi.to(x[0].device)
        mask = mask.to(x[0].device)
        confi = (torch.log(x_att + 1e-9) * self.beta).unsqueeze(0).mul(mask.t().unsqueeze(1))
        confi = confi.max(dim=-1).values
        missing_indices = torch.where(mask == 0)  # (样本索引, 视图索引)
        x_filled = [view.clone() for view in x]  # 复制每个视图的原始特征用于补充
        for sample_idx, view_idx in zip(*missing_indices):
            non_missing_views = mask[sample_idx] == 1  # 找到当前样本其他不缺失的视图
            non_missing_samples = torch.where(mask[:, view_idx] == 1)[0]  # 非缺失样本索引
            if len(non_missing_samples) < top_k:
                top_samples = non_missing_samples  # 如果样本不足 top_k，直接使用所有非缺失样本
            else:
                # 选出注意力分数最高的 top_k 样本
                top_samples = non_missing_samples[
                    torch.topk(x_att_score[view_idx, sample_idx, non_missing_samples], top_k).indices]
            # 从选出的样本中获取特征和注意力得分
            top_features = x[view_idx][top_samples]  # [top_k, d]
            top_scores = x_att_score[view_idx, sample_idx, top_samples]  # [top_k]
            # 加权特征聚合
            score_sum = top_scores.sum()
            weighted_features = (top_features.T * top_scores).sum(dim=1) / (score_sum + (score_sum == 0).float())
            # weighted_features = (top_features.T * top_scores).sum(dim=1) / top_scores.sum()
            x_filled[view_idx][sample_idx] = weighted_features  # 填充缺失的特征
        y_n = self.NN2(y)
        y_n = torch.sigmoid(y_n)

        # 训练模式下应用随机掩码
        if mode == 'train':
            masked_input = []
            prop = 0.15  # 掩码比例
            
            for X in x_filled:
                # 计算要掩码的长度
                mask_len = int(prop * X.size(-1))
                if mask_len < 1:
                    mask_len = 1
                    
                # 处理边界情况
                max_start = max(0, X.size(-1) - mask_len - 1)
                if max_start <= 0:
                    # 特征维度太小，无法应用随机掩码
                    masked_input.append(X)
                    continue
                    
                # 为每个样本生成随机起始位置
                st = torch.randint(low=0, high=max_start, size=(X.size(0),), device=X.device)
                
                # 创建掩码张量
                random_mask = torch.ones_like(X, device=X.device)
                
                # 应用掩码
                for j in range(X.size(0)):
                    end_idx = min(st[j] + mask_len, X.size(-1))
                    random_mask[j, st[j]:end_idx] = 0
                    
                # 应用掩码到数据
                masked_X = X * random_mask
                masked_input.append(masked_X)
            
            # 使用掩码后的数据
            x_filled = masked_input

        return x_filled, y_n