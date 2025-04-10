import torch
import torch.nn as nn 
# from Layers import EncoderLayer, DecoderLayer
# from Embed import Embedder, PositionalEncoder
import random
import copy
import math
import numpy as np
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
from layers_new import  FDModel, MLP,GAT
from model_VAE_new import VAE
from myloss import Loss
from model_VAE_new import gaussian_reparameterization_var

def gaussian_reparameterization_std(means, std, times=1):
    std = std.abs()
    means = torch.nan_to_num(means, nan=0.0)
    var = torch.nan_to_num(var, nan=1e-6)
    var = torch.clamp(var, min=1e-6)  # 确保方差为正
    res = torch.zeros_like(means).to(means.device)
    for t in range(times):
        epi = std.data.new(std.size()).normal_()
        res += epi * std + means
    return res/times

def gaussian_reparameterization_var(means, var, times=1):
    ## 重参数化
    std = torch.sqrt(var+1e-8)
    assert torch.sum(std<0).item()==0
    res = torch.zeros_like(means).to(means.device)
    for t in range(times):
        epi = std.data.new(std.size()).normal_()
        res += epi * std + means
    return res/times

def Init_random_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
def setEmbedingModel(d_list,d_out):
    return nn.ModuleList([Mlp(d,d,d_out)for d in d_list])

def fill_with_label(label_embedding,label,x_embedding,inc_V_ind):
    fea = label.matmul(label_embedding)
    new_x =  x_embedding*inc_V_ind.unsqueeze(-1) + fea.unsqueeze(1)
    return new_x


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=[512,1024,1024,1024,512], act=nn.GELU, norm=nn.BatchNorm1d, final_act=True, final_norm=True):
        super(MLP, self).__init__()
        self.act = act
        self.norm = norm
        # init layers
        self.mlps = []
        layers = []
        if len(hidden_dim) > 0:
            layers.append(nn.Linear(in_dim, hidden_dim[0]))
            layers.append(self.norm(hidden_dim[0]))
            layers.append(self.act())
            self.mlps.append(nn.Sequential(*layers))
            layers = []
            ##hidden layer
            for i in range(len(hidden_dim)-1):
                layers.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))
                layers.append(self.norm(hidden_dim[i+1]))
                layers.append(self.act())
                self.mlps.append(nn.Sequential(*layers))
                layers = []
            ##output layer
            layers.append(nn.Linear(hidden_dim[-1], out_dim))
            if final_norm:
                layers.append(self.norm(out_dim))
            if final_act:
                layers.append(self.act())
            self.mlps.append(nn.Sequential(*layers))
            layers = []
        else:
            layers.append(nn.Linear(in_dim, out_dim))
            if final_norm:
                layers.append(self.norm(out_dim))
            if final_act:
                layers.append(self.act())
            self.mlps.append(nn.Sequential(*layers))
        self.mlps = nn.ModuleList(self.mlps)
        
    def forward(self, x):
        for layers in self.mlps:
            x = layers(x)
        return x

class Qc_inference_mlp(nn.Module):
    # 变分推断部分--encoder，将输入数据映射为高斯分布的参数： 均值mu和尺度sca
    def __init__(self, in_dim, out_dim,hidden_dim=[1024]):
        super(Qc_inference_mlp, self).__init__()
        self.transfer_act = nn.ReLU
        self.mlp = MLP(in_dim, out_dim,hidden_dim=hidden_dim)
        self.z_loc = nn.Linear(out_dim, out_dim)
        self.z_sca = nn.Sequential(nn.Linear(out_dim, out_dim), nn.Softplus())
        # self.qzv_inference = nn.Sequential(*self.qz_layer)
    def forward(self, x):
        assert torch.sum(torch.isnan(x)).item() == 0
        hidden_features = self.mlp(x)
        c_mu = self.z_loc(hidden_features)
        c_sca = self.z_sca(hidden_features)
        # class_feature  = self.z
        if torch.sum(torch.isnan(c_mu)).item() >0:
            pass
        assert torch.sum(torch.isinf(c_mu)).item() == 0
        return c_mu, c_sca

class Net(nn.Module):
    def __init__(self, d_list,num_classes,beta,in_layers,class_emb,adj,rand_seed=0, z_dim=512):
        super(Net, self).__init__()
        # self.configs = configs
        self.rand_seed = rand_seed
        # Label semantic encoding module
        self.d_list = d_list
        # 标签嵌入是一个512维的对角矩阵
        self.label_embedding = nn.Parameter(torch.eye(num_classes),
                                            requires_grad=True)
        self.num_classes = num_classes
        # 标签嵌入的vae版本（即用来计算标签的均值和方差）
        self.label_embedding_u = nn.Parameter(torch.eye(num_classes),
                                            requires_grad=True)
        # 标签相关图也是一个512维的对角矩阵
        self.label_adj = nn.Parameter(torch.eye(num_classes),
                                      requires_grad=True)  #标签相关性
        self.adj = adj
        # z_dim是嵌入的维度
        self.z_dim = z_dim

        self.GAT_encoder = GAT(num_classes, class_emb)
        self.GAT_encoder_vae = GAT(z_dim, class_emb)
        
        # Semantic-guided feature-disentangling module
        self.FD_model = FDModel(d_list,class_emb,
                                512, 512,beta, in_layers, 1,
                                False, 'leaky_relu', 0.1)
        self.label_mlp = Qc_inference_mlp(num_classes, z_dim)
        self.mix_prior = None
        self.mix_mu = None
        self.mix_sca = None
        self.k = num_classes
        # VAE module
        self.VAE = VAE(d_list=self.d_list,z_dim=self.z_dim,class_num=self.num_classes)

        self.batchnorm = nn.BatchNorm1d(z_dim)
        # Classifier
        self.cls_conv = nn.Conv1d(num_classes, num_classes,
                                  512, groups=num_classes)
        self.cls = nn.Linear(z_dim, num_classes)
        hidden_list = [512] * (1-1)
        self.NN3 = MLP(512, 512, hidden_dim=hidden_list, 
                       act=nn.LeakyReLU(0.1), norm=nn.BatchNorm1d, 
                       final_act=False, final_norm=False)
        self.maxP = nn.MaxPool2d((len(d_list),1))
        self.cuda()
        self.view_cls = nn.ModuleList([nn.Linear(z_dim, num_classes) for i in range(len(d_list)) ])
        self.set_prior()
        # 
        self.reset_parameters()

    def set_prior(self):
        # set prior components weights
        # 为混合高斯分布初始化参数
        self.mix_prior = nn.Parameter(torch.full((self.k,), 1 / self.k), requires_grad=True)
        # set prior gaussian components mean
        # self.mix_mu = nn.Parameter(torch.full((self.k, self.z_dim), 0.0), requires_grad=True)
        self.mix_mu = nn.Parameter(torch.rand((self.k,self.z_dim)),requires_grad=True)
        # set prior gaussian components scale
        self.mix_sca = nn.Parameter(torch.rand((self.k,self.z_dim)),requires_grad=True)
  

    def reset_parameters(self):
        Init_random_seed(self.rand_seed)
        nn.init.normal_(self.label_embedding)
        nn.init.normal_(self.label_adj)
        self.FD_model.reset_parameters()
        self.cls_conv.reset_parameters()


        
    def get_config_optim(self):
        return [{'params': self.FD_model.parameters()},
                {'params': self.cls_conv.parameters()}]

    def forward(self, input, mask, inc_L_ind, mode):
        # Generating semantic label embeddings via label semantic encoding module

        # 利用图注意机制计算multi-head output features
        # 对应论文Multi-label Semantic Representation Learning的部分，根据初始的标签邻接矩阵计算出一个标签的嵌入
        # 其中label_embedding features是可以学习的，注意力系数计算中的W，a都是可以学习的参数
        # 此处是vae的标签嵌入
        label_embedding_vae  =  self.label_embedding_u
        label_embedding = self.GAT_encoder(label_embedding_vae, self.adj)
        # 对应论文Attention-Induced Missing View Imputation的部分
        ## 返回值Z对应论文中的公式8返回值：B_i
        x_new_processed, y_n = self.FD_model(input, label_embedding, mask, mode)  #Z[i]=[128, 260, 512] b c d_e
        # 将标签的vae嵌入label_embedding_u（初始化为一个大小为num_classes*num_classes的对角矩阵）输入进变分推断的encoder，得到高斯分布的均值和幅度sca
        label_embedding_mu, label_embedding_var = self.label_mlp(label_embedding_vae)
        label_embedding_mu = self.GAT_encoder_vae(label_embedding_mu, self.adj)
        label_embedding_var = self.GAT_encoder_vae(label_embedding_var, self.adj)
        label_embedding_var = torch.nn.Softplus()(label_embedding_var)
        label_embedding_sample = gaussian_reparameterization_var(label_embedding_mu,label_embedding_var,5)
        assert torch.sum(torch.isnan(label_embedding_sample)).item() == 0
        assert torch.sum(torch.isinf(label_embedding_sample)).item() == 0
        uniview_mu_list, uniview_sca_list, xr_s_list, xr_p_list, pos_beat_I, I_mutual_s,z_sample_list_s, z_sample_list_p, mu_s_list, mu_p_list, sca_s_list, sca_p_list = self.VAE(x_new_processed, mask)
        # 这一步得到的p_pre对应论文中的（公式8）下面的P(v)部分，得到的是经过线性分类层的初次logits
        # 将mu_s_list, mu_p_list, sca_s_list, sca_p_list 转换为tensor,这些是共享和私有的潜在变量
        mu_s_list = torch.stack(mu_s_list,dim=0)
        mu_p_list = torch.stack(mu_p_list,dim=0)
        sca_s_list = torch.stack(sca_s_list,dim=0)
        sca_p_list = torch.stack(sca_p_list,dim=0)
        # p_list = []

        # for z_i in Z:
        #     p_ii = self.cls_conv(z_i).squeeze(2)
        #     p_list.append(p_ii)
        # p_pre = p_list
        # p=torch.stack(p_list,dim=1)        # b*m*c

        # （对应公式10）计算加权的mask
        # 计算加权掩码，并已经进行过了归一化处理
        # mask_confi = (1-mask).mul(confi.t())+mask # b m
        # mask_confi = mask_confi/(mask_confi.sum(dim=1,keepdim=True)+1e-9)   # b*m
        # # 应用权重（对应公式11），用置信度对输出分类进行加权，根据视图缺失情况和邻居相关性动态调整置信度，提升鲁棒性
        # p = p.mul(mask_confi.unsqueeze(dim=-1)) # 乘以权重
        # p = p.sum(dim = 1) # 跨视图聚合
        # p = torch.sigmoid(p) # 转换为概率
        # 这里是vae模块得到的预测即伪标签
        p_vae_s_list = []
        p_vae_p_list = []
        for v in range(len(z_sample_list_s)):
            # 将标签嵌入的采样和潜在变量进行融合的特征
            qc_z_s = (z_sample_list_s[v].unsqueeze(1)) * ((label_embedding_sample).sigmoid().unsqueeze(0))
            qc_z_p = (z_sample_list_p[v].unsqueeze(1)) * ((label_embedding_sample).sigmoid().unsqueeze(0))
            p_vae_s = self.cls_conv(qc_z_s).squeeze(-1)
            p_vae_p = self.cls_conv(qc_z_p).squeeze(-1)
            p_vae_s = torch.sigmoid(p_vae_s)
            p_vae_p = torch.sigmoid(p_vae_p)
            p_vae_s_list.append(p_vae_s)
            p_vae_p_list.append(p_vae_p)
        # 至此得到了每个视图的共享和私有的预测伪标签
        # 我们需要根据这个伪标签和其共享和私有的特征计算流形损失
        loss_mani = Loss()
        # 这里是针对每个视图的共享和私有的表征（z）及其伪标签去计算流形损失以得到融合系数
        loss_manifold_s = []
        loss_manifold_p = []
        for v in range(len(p_vae_s_list)):
            loss_manifold_s.append(loss_mani.simplified_manifold_loss(z_sample_list_s[v], p_vae_s_list[v]))
            loss_manifold_p.append(loss_mani.simplified_manifold_loss(z_sample_list_p[v], p_vae_p_list[v]))

        ## 至此，计算出了每个视图的流形损失，接下来需要利用这个流形损失对每个视图的共享和私有特征进行融合
        loss_manifold_s = torch.tensor(loss_manifold_s)
        loss_manifold_p = torch.tensor(loss_manifold_p)
        # 损失值越小，权重越大,直接使用倒数
        weights_s = 1.0 / (loss_manifold_s + 1e-10)  # 添加小常数避免除零
        weights_p = 1.0 / (loss_manifold_p + 1e-10)
        # 归一化权重，使其和为1
        weights_s = weights_s / (weights_s.sum() + 1e-10)
        weights_p = weights_p / (weights_p.sum() + 1e-10)
        # 将共享和私有的潜在变量进行融合
        aggregate_mu_s, aggregate_sca_s = self.VAE.weighted_poe_aggregate(mu_s_list, sca_s_list, weights_s)
        aggregate_mu_p, aggregate_sca_p = self.VAE.weighted_poe_aggregate(mu_p_list, sca_p_list, weights_p)
        fusion_s = gaussian_reparameterization_var(aggregate_mu_s, aggregate_sca_s, 5)
        fusion_p = gaussian_reparameterization_var(aggregate_mu_p, aggregate_sca_p, 5)
        fusion_p_sigmoid = fusion_p.sigmoid()  # 创建新的张量而不是修改原始张量
        fusion_fea = fusion_s * fusion_p_sigmoid
        # 分别将共享s，和私有p的流形损失求和再平均
        loss_manifold_s_avg = loss_manifold_s.mean()
        loss_manifold_p_avg = loss_manifold_p.mean()
        fusion_fea = self.batchnorm(fusion_fea)
        Z = fusion_fea.clone().unsqueeze(1) * y_n.clone().unsqueeze(0)
        # 对Z进行batchnorm
        pred = self.cls_conv(Z)
        pred = pred.sigmoid().squeeze(2)
        return pred, xr_s_list, xr_p_list, pos_beat_I, p_vae_s_list, p_vae_p_list, I_mutual_s, loss_manifold_s_avg, loss_manifold_p_avg
        

def get_model(d_list,num_classes,beta,in_layers,class_emb,adj,rand_seed=0):
    model = Net(d_list,num_classes=num_classes,beta=beta,in_layers=in_layers,class_emb=class_emb,adj=adj,rand_seed=rand_seed, z_dim=512)
    model = model.to(torch.device('cuda' if torch.cuda.is_available() 
                                    else 'cpu'))
    return model

if __name__=="__main__":
    # input=torch.ones([2,10,768])
    from MLdataset import getIncDataloader
    dataloder,dataset = getIncDataloader('/disk/MATLAB-NOUPLOAD/MyMVML-data/corel5k/corel5k_six_view.mat','/disk/MATLAB-NOUPLOAD/MyMVML-data/corel5k/corel5k_six_view_MaskRatios_0_LabelMaskRatio_0_TraindataRatio_0.7.mat',training_ratio=0.7,mode='train',batch_size=3,num_workers=2)
    input = next(iter(dataloder))[0]
    model=get_model(num_classes=260,beta=0.2,in_features=1,class_emb=260,rand_seed=0)
    input = [v_data.to('cuda:0') for v_data in input]
    # print(input[0])
    pred,_,_=model(input)
