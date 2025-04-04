import torch
import torch.nn as nn
import torch.nn.functional as F

def cosdis(x1,x2):
    return (1-torch.cosine_similarity(x1,x2,dim=-1))/2
class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
    def emb_loss(self, emb, adj):
        '''
        Parameters
        ----------
        emb : Tensor
            An MxE tensor, the embedding of the ith node is stored in emb[i,:].
        adj : Tensor
            An MxM tensor, adjacent matrix of the graph.
        
        Returns
        -------
        loss : float
            The link prediction loss.
        '''
        # print()
        emb_norm = emb.norm(dim=1, keepdim=True)
        emb_norm = emb / (emb_norm.cuda() + 1e-6)
        adj_pred = torch.matmul(emb_norm, emb_norm.t())
        adj_pred = (torch.matmul(emb_norm, emb_norm.t())+1)/2
        adj_pred = torch.clamp(adj_pred,min=0.,max=1.)
        loss = torch.mean(-adj.mul((adj_pred+1e-9).log())-(1-adj).mul((1-adj_pred+1e-5).log()))
        # print(torch.isnan((1-adj_pred+1e-5).log()).sum())
        # loss = torch.mean(torch.pow(adj - adj_pred, 2))
        return loss
    def rec_loss(self,recx,orix,mask):
        mask_mn1 = mask.t().unsqueeze(-1)
        loss = torch.pow((recx - orix), 2).mul(mask_mn1).mean()
        return loss
    def weighted_wmse_loss(self,input, target, weight, reduction='mean'):
        ret = (torch.diag(weight).mm(target - input)) ** 2
        ret = torch.mean(ret)
        return ret

    def weighted_BCE_loss(self,target_pre,sub_target,inc_L_ind,reduction='mean'):
        ## 该loss是缺失标签情况下的加权分类loss
        assert torch.sum(torch.isnan(torch.log(target_pre))).item() == 0
        assert torch.sum(torch.isnan(torch.log(1 - target_pre + 1e-5))).item() == 0
        res=torch.abs((sub_target.mul(torch.log(target_pre + 1e-5)) \
                                                + (1-sub_target).mul(torch.log(1 - target_pre + 1e-5))).mul(inc_L_ind))
        if reduction=='mean':
            return torch.sum(res)/torch.sum(inc_L_ind)
        elif reduction=='sum':
            return torch.sum(res)
        elif reduction=='none':
            return res
                            
    def BCE_loss(self,target_pre,sub_target):
        return torch.mean(torch.abs((sub_target.mul(torch.log(target_pre + 1e-10)) \
                                        + (1-sub_target).mul(torch.log(1 - target_pre + 1e-10)))))
    
    def label_guided_graph_single_loss(self, x, inc_labels, inc_V_ind, inc_L_ind):
        """
        计算基于标签引导的图损失
        
        参数:
        - x: 形状为(n,d)的张量，表示n个样本的d维特征
        - inc_labels: 形状为(n,c)的张量，表示n个样本的c维标签
        - inc_V_ind: 形状为(n,)的张量，表示视图掩码
        - inc_L_ind: 形状为(n,c)的张量，表示标签掩码
        
        返回:
        - loss: 标量，表示损失值
        """
        n = x.size(0)  # 样本数量
        if n == 1:
            return 0
        
        valid_labels_sum = torch.matmul(inc_L_ind.float(), inc_L_ind.float().T)  # [n, n]
        
        # 计算标签相似度矩阵T
        labels = (torch.matmul(inc_labels, inc_labels.T) / (valid_labels_sum + 1e-9)).fill_diagonal_(0)
        
        # 计算特征相似度
        x = F.normalize(x, p=2, dim=1)  # 在特征维度上归一化
        cos_sim = torch.matmul(x, x.T)  # [n, n]
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
        sim = (1 + cos_sim) / 2  # [n, n]
        
        # 创建掩码，排除自身与自身的比较
        mask = (inc_V_ind.sum(dim=1) > 0).float().unsqueeze(1).mul((inc_V_ind.sum(dim=1) > 0).float().unsqueeze(0))  # [n, n]
        mask = mask.masked_fill(torch.eye(n, device=x.device)==1, 0.)
        
        assert torch.sum(torch.isnan(mask)).item() == 0
        assert torch.sum(torch.isnan(labels)).item() == 0
        assert torch.sum(torch.isnan(sim)).item() == 0
        
        # 计算损失
        loss = self.weighted_BCE_loss(
            sim.view(-1),  # 展平为一维
            labels.view(-1),  # 展平为一维
            mask.view(-1),  # 展平为一维
            reduction='none'
        )
        
        # 归一化损失
        loss = loss.sum() / (mask.sum() + 1e-9)
        return 0.5 * loss
    
    def label_guided_graph_loss(self, x, inc_labels, inc_V_ind, inc_L_ind):
        n = x.size(1)
        v = x.size(0)
        if n == 1:
            return 0
        valid_labels_sum = torch.matmul(inc_L_ind.float(), inc_L_ind.float().T) #[n, n] 

        ## 计算T 对应（2）(对应文章当中的公式2，即标签相似度矩阵T)
        labels = (torch.matmul(inc_labels, inc_labels.T) / (valid_labels_sum + 1e-9)).fill_diagonal_(0)


        # labels = torch.softmax(labels.masked_fill(labels==0,-1e9),dim=-1)，此处已经归一化了
        x = F.normalize(x, p=2, dim=-1)
        x_T = torch.transpose(x,-1,-2)#[v,d,n]
        cos_sim = torch.matmul(x, x_T)
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)  # 强制限制余弦相似度
        sim = (1 + cos_sim) / 2 # [v, n, n]

        mask_v = (inc_V_ind.T).unsqueeze(-1).mul((inc_V_ind.T).unsqueeze(1)) #[v, n, n]
        mask_v = mask_v.masked_fill(torch.eye(n,device=x.device)==1,0.)
        assert torch.sum(torch.isnan(mask_v)).item() == 0
        assert torch.sum(torch.isnan(labels)).item() == 0
        assert torch.sum(torch.isnan(sim)).item() == 0

        # print('labels',torch.sum(torch.max(labels)))
        # loss = ((sim.view(v,-1)-labels.view(1,n*n))**2).mul(mask_v.view(v,-1)) # sim labels view [v, n* n]
        # sim = torch.clamp(sim, 0, 1) # 进行数值的裁剪
        loss = self.weighted_BCE_loss(sim.view(v,-1),labels.view(1,n*n).expand(v,-1),mask_v.view(v,-1),reduction='none')
        # assert torch.sum(torch.isnan(loss)).item() == 0
        loss =loss.sum(dim=-1)/(mask_v.view(v,-1).sum(dim=-1))
        return 0.5*loss.sum()/v
    
    def z_c_loss_new(self, z_mu, label,  c_mu, inc_L_ind):
        ## 潜在变量和标签对齐loss，对应的是（公式10）
        ## 通过最小化样本特征与对应标签原型均值的距离，使得同类样本在潜在空间中聚集到其标签原型周围。
        # 输入的z_mu：z_sample --> 交叉视图表示z_0
        # 输入的label：label --> 正标签集合 C
        # 输入的c_mu： label_emb_sample --> 标签原型几何l_0
        label_inc = label.mul(inc_L_ind) # 经过掩码之后的不完整标签
        sample_label_emb = (label_inc.matmul(c_mu))/(label_inc.sum(-1)+1e-9).unsqueeze(-1) # 这一步是计算正标签对应原型的加权平均，得到正标签的几何中心
        loss = ((z_mu-sample_label_emb)**2) # 潜在特征和融合之后的标签原型要接近！
        # print(loss.mean())
        return loss.mean()

    