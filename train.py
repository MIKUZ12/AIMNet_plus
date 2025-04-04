import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import os.path as osp
import utils
from utils import AverageMeter
import MLdataset
import argparse
import time
from model_new import get_model
import evaluation
import torch
import numpy as np
import copy
from myloss import Loss
from torch import nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CosineAnnealingLR


def train(loader, model, loss_model, opt, sche, epoch, dep_graph, logger):
    ## 待解决：
    ## 1. model的构造
    ## 2. loader -- dataset 的 getitem 返回值
    ## 3. loss_model
    ## 4. opt -- optimizer
    ## 5. sche -- scheduler
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    mce = nn.MultiLabelSoftMarginLoss()
    model.train()
    end = time.time()
    for i, (data, label, inc_V_ind, inc_L_ind) in enumerate(loader):
        data_time.update(time.time() - end)
        data = [v_data.to('cuda:0') for v_data in data]
        # sim_based_label = getMvKNNGraph(label)
        label = label.to('cuda:0')
        inc_V_ind = inc_V_ind.float().to('cuda:0')
        inc_L_ind = inc_L_ind.float().to('cuda:0')
        # print(label.shape,label.sum())
        L_all = []
        # pred, label_embedding,  p_pre = model(data,mask=torch.ones_like(inc_V_ind))
        (pred, label_embedding, p_pre, complete_z, uniview_mu_list, uniview_sca_list, 
         label_embedding_sample, p, label_embedding_vae, 
        label_embedding_var, _, xr_s_list, xr_p_list, pos_beat_I, 
        p_vae_s_list, p_vae_p_list, I_mutual_s) = model(data, mask=inc_V_ind, inc_L_ind=inc_L_ind)

        loss_vae_s = 0
        loss_vae_p = 0
        for v in range(len(p_vae_s_list)):
            loss_vae_s += loss_model.weighted_BCE_loss(p_vae_s_list[v], label, inc_L_ind)
            loss_vae_p += loss_model.weighted_BCE_loss(p_vae_p_list[v], label, inc_L_ind)
        # 使用weighted_BCE_loss + label_guided_graph_loss作为损失，使用加权的多label分类loss，输入的值有label的掩码
        loss_CL = loss_model.weighted_BCE_loss(pred, label, inc_L_ind)
        # 这里的输入是：补全之后的多视图特征，形状是(6,128,512),label (128, 20)
        loss_GC = loss_model.label_guided_graph_loss(complete_z, label, inc_V_ind, inc_L_ind)
        ## 剩下的loss还没有引入train中，但是都已实现完
        loss = loss_CL + loss_GC + loss_vae_s + loss_vae_p
        opt.zero_grad()
        loss.backward()
        if isinstance(sche, CosineAnnealingWarmRestarts):
            sche.step(epoch + i / len(loader))
        opt.step()

        losses.update(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()
    if isinstance(sche, StepLR):
        sche.step()
    logger.info('Epoch:[{0}]\t'
                'Time {batch_time.avg:.3f}\t'
                'Data {data_time.avg:.3f}\t'
                'Loss {losses.avg:.3f}'.format(
        epoch, batch_time=batch_time,
        data_time=data_time, losses=losses))
    return losses, model


def test(loader, model, loss_model, epoch, logger):
    batch_time = AverageMeter()
    losses = AverageMeter()
    total_labels = []
    total_preds = []
    model.eval()
    end = time.time()
    for i, (data, label, inc_V_ind, inc_L_ind) in enumerate(loader):
        # data_time.update(time.time() - end)
        data = [v_data.to('cuda:0') for v_data in data]
        # pred,_,_ = model(data,mask=torch.ones_like(inc_V_ind).to('cuda:0'))
        (pred, label_embedding, p_pre, complete_z, uniview_mu_list, uniview_sca_list, 
         label_embedding_sample, p, label_embedding_vae, 
        label_embedding_var, _, xr_s_list, xr_p_list, pos_beat_I, 
        p_vae_s_list, p_vae_p_list, I_mutual_s) = model(data, mask=inc_V_ind, inc_L_ind=inc_L_ind)
        pred = pred.cpu()
        total_labels = np.concatenate((total_labels, label.numpy()), axis=0) if len(total_labels) > 0 else label.numpy()
        total_preds = np.concatenate((total_preds, pred.detach().numpy()), axis=0) if len(
            total_preds) > 0 else pred.detach().numpy()
        loss = loss_model.weighted_BCE_loss(pred, label, inc_L_ind)
        losses.update(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()
    total_labels = np.array(total_labels)
    total_preds = np.array(total_preds)
    evaluation_results = evaluation.do_metric(total_preds, total_labels)
    logger.info('Epoch:[{0}]\t'
                'Time {batch_time.avg:.3f}\t'
                'Loss {losses.avg:.3f}\t'
                'AP {ap:.3f}\t'
                'HL {hl:.3f}\t'
                'RL {rl:.3f}\t'
                'AUC {auc:.3f}\t'.format(
        epoch, batch_time=batch_time,
        losses=losses,
        ap=evaluation_results[4],
        hl=evaluation_results[0],
        rl=evaluation_results[3],
        auc=evaluation_results[5]
    ))
    return evaluation_results


def main(args, file_path):
    panduan_list = [4]
    for panduan in panduan_list:
    #     if panduan == 5:
    #         args.label_missing_rates = [0]
    #         args.feature_missing_rates = [0]
    #         args.folds_num = 10
    #     if panduan == 0:
    #         args.label_missing_rates = [0.5]
    #         args.feature_missing_rates = [0, 0.1, 0.3, 0.7]
    #         args.folds_num = 3
    #     if panduan == 1:
    #         args.label_missing_rates = [0, 0.1, 0.3, 0.7]
    #         args.feature_missing_rates = [0.5]
    #         args.folds_num = 3
    #     if panduan == 2:
    #         args.label_missing_rates = [0.1]
    #         args.feature_missing_rates = [0.1]
    #         args.folds_num = 3
    #     if panduan == 3:
    #         args.label_missing_rates = [0.3]
    #         args.feature_missing_rates = [0.3]
    #         args.folds_num = 3
        if panduan == 4:
            args.label_missing_rates = [0.5]
            args.feature_missing_rates = [0.5]
            args.folds_num = 3
        for label_missing_rate in args.label_missing_rates:
            for feature_missing_rate in args.feature_missing_rates:
                ## 先迭代label的缺失，再迭代feature的缺失
                folds_num = args.folds_num
                test_acc_list = []
                test_one_error_list = []
                test_coverage_list = []
                test_ranking_loss_list = []
                test_precision_list = []
                test_auc_list = []
                test_f1_list = []
                training_time_list = []
                for fold_idx in range(folds_num):
                    ## 折验证的迭代。以下是一些路径的定义
                    seed = fold_idx
                    # 数据集的名称，直接在add purse中定义
                    dataset_name = args.dataset
                    # 折验证的根目录
                    folder_name = os.path.join('E:\\desktop\\AIMNet\\Index', dataset_name)
                    # 拆分数据集的名称
                    file_name = f'{dataset_name}_label_missing_rate_{label_missing_rate}_feature_missing_rate_{feature_missing_rate}_seed_{seed + 1}.mat'
                    # 原始数据集的路径
                    data_path = osp.join(args.root_dir, args.dataset + '.mat')
                    # fold_data_path = osp.join(args.root_dir, args.dataset, args.dataset + str(
                    #     mask_label_ratio) + '_LabelMaskRatio_' + str(mask_view_ratio) + '_MaskRatios_' + '.mat')
                    index_file_path = os.path.join(folder_name, file_name)
                    fold_data_path = index_file_path
                    folds_results = [AverageMeter() for i in range(9)]
                    # 设置日志路径
                    if args.logs:
                        logfile = osp.join(args.logs_dir, args.name + args.dataset + '_V_' + str(
                            args.mask_view_ratio) + '_L_' +
                                           str(args.mask_label_ratio) + '_T_' +
                                           str(args.training_sample_ratio) + '_' + str(args.alpha) + '_' + str(
                            args.beta) + '.txt')
                    else:
                        logfile = None
                    logger = utils.setLogger(logfile)
                    # 读取数据的dataloader
                    train_dataloder, train_dataset = MLdataset.getIncDataloader(data_path, fold_data_path,
                                                                                training_ratio=args.training_sample_ratio,
                                                                                fold_idx=fold_idx, mode='train',
                                                                                batch_size=args.batch_size,
                                                                                shuffle=True, num_workers=0)
                    test_dataloder, test_dataset = MLdataset.getIncDataloader(data_path, fold_data_path,
                                                                              training_ratio=args.training_sample_ratio,
                                                                              val_ratio=0.15, fold_idx=fold_idx,
                                                                              mode='test', batch_size=args.batch_size,
                                                                              num_workers=0)
                    val_dataloder, val_dataset = MLdataset.getIncDataloader(data_path, fold_data_path,
                                                                            training_ratio=args.training_sample_ratio,
                                                                            fold_idx=fold_idx, mode='val',
                                                                            batch_size=args.batch_size, num_workers=0)
                    
                    d_list = train_dataset.d_list
                    classes_num = train_dataset.classes_num
                    labels = torch.tensor(train_dataset.cur_labels).float().to('cuda:0')
                    # 获得label correlation matrix
                    ## 标签矩阵的转置 * 标签矩阵 （对应公式1）
                    dep_graph = torch.matmul(labels.T, labels)
                    dep_graph = dep_graph / (torch.diag(dep_graph).unsqueeze(1) + 1e-10)
                    dep_graph[dep_graph <= args.sigma] = 0.
                    dep_graph.fill_diagonal_(fill_value=0.) # 对角线元素置0
                    start = time.time()
                    ## 这里需要修改
                    model = get_model(d_list, num_classes=classes_num, beta=args.beta, in_layers=1, class_emb=512,
                                      adj=dep_graph, rand_seed=0)
                    # print(model)
                    loss_model = Loss()
                    # crit = nn.BCELoss()
                    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9)
                    # optimizer = Adam(model.parameters(), lr=args.lr)
                    # scheduler = StepLR(optimizer, step_size=5, gamma=0.85)
                    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=4, T_mult=2)
                    scheduler = None
                    logger.info('train_data_num:' + str(len(train_dataset)) + '  test_data_num:' + str(
                        len(test_dataset)) + '   fold_idx:' + str(fold_idx))

                    static_res = 0
                    epoch_results = [AverageMeter() for i in range(9)]
                    total_losses = AverageMeter()
                    train_losses_last = AverageMeter()
                    best_epoch = 0
                    best_model_dict = {'model': model.state_dict(), 'epoch': 0}
                    for epoch in range(args.epochs):
                        tt = time.time()
                        ## 这里需要修改train函数
                        train_losses, model = train(train_dataloder, model, loss_model, optimizer, scheduler, epoch,
                                                    dep_graph, logger)
                        # print("traintime:",time.time()-tt)
                        # test_results = test(test_dataloder,model,loss_model,epoch,dep_graph,logger)
                        # tt=time.time()
                        val_results = test(val_dataloder, model, loss_model, epoch, logger)
                        # print("testtime:",time.time()-tt)
                        # for i,re in enumerate(epoch_results):
                        # re.update(test_results[i])
                        if val_results[4] * 0.25 + val_results[3] * 0.25 + val_results[5] * 0.5 >= static_res:
                            static_res = val_results[4] * 0.25 + val_results[3] * 0.25 + val_results[5] * 0.5
                            best_model_dict['model'] = copy.deepcopy(model.state_dict())
                            best_model_dict['epoch'] = epoch
                            best_epoch = epoch
                        train_losses_last = train_losses
                        total_losses.update(train_losses.sum)
                    model.load_state_dict(best_model_dict['model'])
                    end = time.time()
                    print("epoch", best_model_dict['epoch'])
                    test_results = test(test_dataloder, model, loss_model, epoch, logger)
                    test_acc_list.append(test_results[0])
                    test_one_error_list.append(test_results[1])
                    test_coverage_list.append(test_results[2])
                    test_ranking_loss_list.append(test_results[3])
                    test_precision_list.append(test_results[4])
                    test_auc_list.append(test_results[5])
                    test_f1_list.append(test_results[6])
                    training_time_list.append(end - start)
                if 1 + 1 == 2:
                    # 修改保存路径的部分
                    saved_path = f"./test_results/{dataset_name}/"
                    if not os.path.exists(saved_path):
                        os.makedirs(saved_path)
                    # 修改后的文件名为 missing_rate_i{missing_rate_i}_positive_noise_rate_{positive_noise_rate}_negative_noise_rate_{negative_noise_rate}.txt
                    saved_path = saved_path + f"_gamma_{args.gamma}_alpha_{args.alpha}_label_missing_rate_{label_missing_rate}_feature_missing_rate_{feature_missing_rate}.txt"
                    with open(saved_path, 'w') as file:
                        file.write(
                            '============================ Summary Board ============================\n')
                        file.write('>>>>>>>> Experimental Setup:\n')
                        file.write('>>>>>>>> Dataset Info:\n')
                        file.write(f'Dataset name: {dataset_name}\n')
                        file.write('>>>>>>>> Experiment Results:\n')
                        for i in range(1, len(test_acc_list) + 1):
                            file.write(f'Test{i}:  ')
                            file.write(
                                'Accuracy:{0:.4f} | one_error:{1:.4f}| coverage:{2:.4f} | ranking_loss:{3:.4f}| Precision:{4:.4f} | AUC:{5:.4f} | F1:{6:.4f} | \n'.format(
                                    test_acc_list[i - 1], test_one_error_list[i - 1],
                                    test_coverage_list[i - 1],
                                    test_ranking_loss_list[i - 1], test_precision_list[i - 1],
                                    test_auc_list[i - 1], test_f1_list[i - 1]))
                        file.write('>>>>>>>> Mean Score (standard deviation):\n')
                        file.write(
                            'Accuracy: {0:.4f} ({1:.4f})\n'.format(np.mean(test_acc_list),
                                                                   np.std(test_acc_list)))
                        file.write('one_error: {0:.4f} ({1:.4f})\n'.format(
                            np.mean(test_one_error_list),
                            np.std(test_one_error_list)))
                        file.write(
                            'coverage: {0:.4f} ({1:.4f})\n'.format(np.mean(test_coverage_list),
                                                                   np.std(test_coverage_list)))
                        file.write('ranking_loss: {0:.4f} ({1:.4f})\n'.format(
                            np.mean(test_ranking_loss_list),
                            np.std(test_ranking_loss_list)))
                        file.write('Precision: {0:.4f} ({1:.4f})\n'.format(
                            np.mean(test_precision_list),
                            np.std(test_precision_list)))
                        file.write(
                            'AUC: {0:.4f} ({1:.4f})\n'.format(np.mean(test_auc_list),
                                                              np.std(test_auc_list)))
                        file.write(
                            'F1: {0:.4f} ({1:.4f})\n'.format(np.mean(test_f1_list),
                                                             np.std(test_f1_list)))
                        file.write(
                            '>>>>>>>> Mean Training Time (standard deviation)(s):{0:.4f} ({1:.4f})\n'.format(
                                np.mean(training_time_list), np.std(training_time_list)))
                        for itr in range(len(training_time_list)):
                            file.write(f"{itr}: {training_time_list[itr]}\n")
                    file.close()


def filterparam(file_path, index):
    params = []
    if os.path.exists(file_path):
        file_handle = open(file_path, mode='r')
        lines = file_handle.readlines()
        lines = lines[1:] if len(lines) > 1 else []
        params = [[float(line.split(' ')[idx]) for idx in index] for line in lines]
    return params


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--logs', default=False, type=bool)
    parser.add_argument('--records-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'new_records'))
    parser.add_argument('--file-path', type=str, metavar='PATH',
                        default='')
    parser.add_argument('--root-dir', type=str, metavar='PATH',
                        default='E:/desktop/AIMNet/data')
    parser.add_argument('--dataset', type=str,
                        default='')  # mirflickr corel5k pascal07 iaprtc12 espgame,'yeast','pascal07_six_view','mirflickr_six_view','iaprtc12_six_view','espgame_six_view'
    parser.add_argument('--datasets', type=list, default=['pascal07_six_view']) #corel5k_six_view 120,
    parser.add_argument('--feature_missing_rates', type=list, default=[0.5])
    parser.add_argument('--label_missing_rates', type=float, default=[0.5])
    parser.add_argument('--training-sample-ratio', type=float, default=0.7)
    parser.add_argument('--folds-num', default=10, type=int)
    parser.add_argument('--weights-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'weights'))
    parser.add_argument('--curve-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'curves'))
    parser.add_argument('--save-curve', default=False, type=bool)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--name', type=str, default='10_final_')
    # Optimization args
    parser.add_argument('--lr', type=float, default=1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=30)  # set 10 for mirflickr pascal07  set 100 for other datasets
    # Training args
    parser.add_argument('--n_z', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--alpha', type=float, default=1e-1)
    parser.add_argument('--beta', type=float, default=1e-1)
    parser.add_argument('--gamma', type=float, default=1e-1)
    parser.add_argument('--sigma', type=float, default=0.)
    args = parser.parse_args()
    if args.logs:
        if not os.path.exists(args.logs_dir):
            os.makedirs(args.logs_dir)
    if args.save_curve:
        if not os.path.exists(args.curve_dir):
            os.makedirs(args.curve_dir)
    if True:
        if not os.path.exists(args.records_dir):
            os.makedirs(args.records_dir)
    lr_list = [1e0]
    beta_list = [0.2]  # beta is tau in the paper
    sigma_list = [0]
    if args.lr >= 0.01:
        args.momentumkl = 0.90
    for lr in lr_list:
        args.lr = lr
        for beta in beta_list:
            args.beta = beta
            for sigma in sigma_list:
                args.sigma = sigma
                for dataset in args.datasets:
                    args.dataset = dataset
                    file_path = osp.join(args.records_dir, args.name + args.dataset + '_VM_' + str(
                        0.5) + '_LM_' +
                                         str(0.5) + '_T_' +
                                         str(args.training_sample_ratio) + '.txt')
                    args.file_path = file_path
                    existed_params = filterparam(file_path, [-3, -2, -1])
                    # if [args.beta, args.sigma] in existed_params:
                    #     print('existed param! beta:{}'.format(args.beta, args.sigma))
                    #     # continue
                    main(args, file_path)
