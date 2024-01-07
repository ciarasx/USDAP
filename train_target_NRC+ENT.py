import os
import faiss
import torch
import shutil 
import numpy as np
# from torchvision import transforms
# from timm.data.auto_augment import rand_augment_transform
# from helper.data_list import ImageList, ImageList_idx

from tqdm import tqdm 
from model.SFUniDA import SFUniDA
from dataset.dataset import SFUniDADataset
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
# from model import prompters
# from model.mix_prompt import mix_data_prompt
# from bn_loss import bn_loss

from config.model_config import build_args
from typing import Any

from utils.net_utils import set_logger, set_random_seed
from utils.net_utils import compute_h_score, Entropy

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


# def get_strong_aug(dataset, idx):
#     aug_img = torch.cat([dataset[i][0].unsqueeze(dim=0) for i in idx], dim=0)
#     # aug_img.squeeze(0)
#     # print (aug_img.shape)
#     return aug_img
#
# def strong_augment(resize_size=256, crop_size=224, alexnet=False):
#     if not alexnet:
#         normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                          std=[0.229, 0.224, 0.225])
#     else:
#         normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
#     return transforms.Compose([
#         transforms.Resize((resize_size, resize_size)),
#         transforms.RandomCrop(crop_size),
#         rand_augment_transform(config_str='rand-m9-mstd0.5',hparams={'translate_const': 117}),
#         transforms.ToTensor(),
#         normalize
#     ])
#
# def image_train(resize_size=256, crop_size=224, alexnet=False):
#     if not alexnet:
#         normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                          std=[0.229, 0.224, 0.225])
#     else:
#         normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
#     return transforms.Compose([
#         transforms.Resize((resize_size, resize_size)),
#         transforms.RandomCrop(crop_size),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         normalize
#     ])
#
# def image_test(resize_size=256, crop_size=224, alexnet=False):
#     if not alexnet:
#         normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                          std=[0.229, 0.224, 0.225])
#     else:
#         normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
#     return transforms.Compose([
#         transforms.Resize((resize_size, resize_size)),
#         transforms.CenterCrop(crop_size),
#         transforms.ToTensor(),
#         normalize
#     ])
#
# def data_load(args):
#     ## prepare data
#     dsets = {}
#     dset_loaders = {}
#     train_bs = args.batch_size
#     txt_test = open(os.path.join(args.target_data_dir, "image_unida_list.txt"), "r").readlines()
#
#     dsets["train"] = ImageList_idx(txt_test, transform=image_train())
#     dset_loaders["train"] = DataLoader(dsets["train"], batch_size=train_bs, shuffle=True, num_workers=args.num_workers,
#                                       drop_last=True)
#     dsets["strong_aug"] = ImageList_idx(txt_test, transform=strong_augment())
#     dset_loaders["strong_aug"] = DataLoader(dsets["strong_aug"], batch_size=train_bs, shuffle=True, num_workers= args.num_workers, drop_last=True)
#     return dset_loaders, dsets

# def train_prompt(args, model, pretrained_params, train_dataloader, test_dataloader, optimizer, epoch_idx=0.0):
#     model.eval()
#     # hard_psd_label_bank, pred_cls_bank, embed_feat_bank = obtain_global_pseudo_labels(args, model, test_dataloader,
#     #                                                                                   epoch_idx)
#     model.train()
#     #
#     # local_KNN = args.local_K
#     # all_pred_loss_stack = []
#     # psd_pred_loss_stack = []
#     pls_pred_loss_stack = []
#     #
#     iter_idx = epoch_idx * len(train_dataloader)
#     iter_max = args.epochs * len(train_dataloader)
#     #
#     for imgs_train, _, _, imgs_idx in tqdm(train_dataloader, ncols=60):
#         iter_idx += 1
#         imgs_idx = imgs_idx.cuda()
#         imgs_train = imgs_train.cuda()
#
#         # hard_psd_label = hard_psd_label_bank[imgs_idx]  # [B, C]
#
#         embed_feat, pred_cls, bn_f, data_prompt = model.forward_pls(imgs_train, apply_softmax=True)
#
#         loss = bn_loss(model, pretrained_params, bn_f, alpha=0.01, i=3)
#     #
#     #     psd_pred_loss = torch.sum(-hard_psd_label * torch.log(pred_cls + 1e-5), dim=-1).mean()
#     #
#     #     with torch.no_grad():
#     #         embed_feat = embed_feat / torch.norm(embed_feat, p=2, dim=-1, keepdim=True)
#     #         feat_dist = torch.einsum("bd, nd -> bn", embed_feat, embed_feat_bank)  # [B, N]
#     #         nn_feat_idx = torch.topk(feat_dist, k=local_KNN + 1, dim=-1, largest=True)[-1]  # [B, local_KNN+1]
#     #         nn_feat_idx = nn_feat_idx[:, 1:]  # [B, local_KNN]
#     #         nn_pred_cls = torch.mean(pred_cls_bank[nn_feat_idx], dim=1)  # [B, C]
#     #         # update the pred_cls and embed_feat bank
#     #         pred_cls_bank[imgs_idx] = pred_cls
#     #         embed_feat_bank[imgs_idx] = embed_feat
#     #
#     #     knn_pred_loss = torch.sum(-nn_pred_cls * torch.log(pred_cls + 1e-5), dim=-1).mean()
#     #
#     #     loss = args.lam_psd * psd_pred_loss + args.lam_knn * knn_pred_loss
#     #
#         lr_scheduler(optimizer, iter_idx, iter_max)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     #
#     #     all_pred_loss_stack.append(loss.cpu().item())
#     #     psd_pred_loss_stack.append(psd_pred_loss.cpu().item())
#         pls_pred_loss_stack.append(loss.cpu().item())
#     #
#     train_loss_dict = {}
#     # train_loss_dict["all_pred_loss"] = np.mean(all_pred_loss_stack)
#     # train_loss_dict["psd_pred_loss"] = np.mean(psd_pred_loss_stack)
#     train_loss_dict["loss"] = np.mean(pls_pred_loss_stack)
#
#     return train_loss_dict, data_prompt

best_score = 0.0
best_coeff = 1.0

@torch.no_grad()
def obtain_global_pseudo_labels(args, model, data_prompt, dataloader, epoch_idx=0.0):
    model.eval()

    pred_cls_bank = [] 
    gt_label_bank = []
    embed_feat_bank = []
    class_list = args.target_class_list
    # dset_loaders, dsets = data_load(args)
    
    args.logger.info("Generating one-vs-all global clustering pseudo labels...")
    
    for _, imgs_test, imgs_label, imgs_idx in tqdm(dataloader, ncols=60):
        
        imgs_test = imgs_test.cuda()
        # inputs_test_stg = get_strong_aug(dsets["strong_aug"], imgs_idx)
        # inputs_test_wk = imgs_train.cuda()
        # imgs_test = inputs_test_stg.cuda()
        # embed_feat, pred_cls = model(imgs_test, apply_softmax=True)
        embed_feat, pred_cls = model.forward_pls(imgs_test, data_prompt, apply_softmax=True)
        pred_cls_bank.append(pred_cls)
        embed_feat_bank.append(embed_feat)
        gt_label_bank.append(imgs_label.cuda())
    
    pred_cls_bank = torch.cat(pred_cls_bank, dim=0) #[N, C]
    gt_label_bank = torch.cat(gt_label_bank, dim=0) #[N]
    embed_feat_bank = torch.cat(embed_feat_bank, dim=0) #[N, D]
    embed_feat_bank = embed_feat_bank / torch.norm(embed_feat_bank, p=2, dim=1, keepdim=True)
    
    global best_score
    global best_coeff
    # At the first epoch, we need to determine the number of categories in target domain, i.e., the C_t in our paper.
    # Here, we utilize the Silhouette metric to realize this goal.
    if epoch_idx == 0.0:
        embed_feat_bank_cpu = embed_feat_bank.cpu().numpy()
        
        # 对嵌入层特征随机采样
        if args.dataset == "VisDA" or args.dataset == "DomainNet":
            # np.random.seed(2021)
            data_size = embed_feat_bank_cpu.shape[0]
            sample_idxs = np.random.choice(data_size, data_size//3, replace=False)
            embed_feat_bank_cpu = embed_feat_bank_cpu[sample_idxs, :]
            
        # 用t-SNE算法降维到二维
        embed_feat_bank_cpu = TSNE(n_components=2, init="pca", random_state=0).fit_transform(embed_feat_bank_cpu)
        # 系数列表
        coeff_list = [0.25, 0.50, 1, 1.05, 2, 3]
        
        for coeff in coeff_list:
            # 计算聚类数目（目标域类数）
            KK = max(int(args.class_num * coeff), 2)
            # k聚类
            kmeans = KMeans(n_clusters=KK, random_state=0).fit(embed_feat_bank_cpu)
            cluster_labels = kmeans.labels_
            # 轮廓系数
            sil_score = silhouette_score(embed_feat_bank_cpu, cluster_labels)
            
            #根据轮廓系数确定最佳系数
            if sil_score > best_score:
                best_score = sil_score
                best_coeff = coeff
    
    KK = int(args.class_num * best_coeff)
    
    data_num = pred_cls_bank.shape[0]
    # 确定目标域每一类的样本数：样本总数/（源类数*系数）
    pos_topk_num = int(data_num / args.class_num / best_coeff)
    # 对于每个类别对样本进行排序，pred_cls_bank是一个[N,C]的张量，N表示样本数，C表示类别数,按照列对pred_cls_bank进行降序排序，返回排序值和对应索引
    sorted_pred_cls, sorted_pred_cls_idxs = torch.sort(pred_cls_bank, dim=0, descending=True)
    # 选择每个类别的前k个正样本和负样本
    pos_topk_idxs = sorted_pred_cls_idxs[:pos_topk_num, :].t() #[C, pos_topk_num]
    neg_topk_idxs = sorted_pred_cls_idxs[pos_topk_num:, :].t() #[C, neg_topk_num]
    
    # 扩展维度，方便计算
    pos_topk_idxs = pos_topk_idxs.unsqueeze(2).expand([-1, -1, args.embed_feat_dim]) #[C, pos_topk_num, D]
    neg_topk_idxs = neg_topk_idxs.unsqueeze(2).expand([-1, -1, args.embed_feat_dim]) #[C, neg_topk_num, D]
    
    embed_feat_bank_expand = embed_feat_bank.unsqueeze(0).expand([args.class_num, -1, -1]) #[C, N, D]
    # 选择每个类别前k个正样本的嵌入特征
    pos_feat_sample = torch.gather(embed_feat_bank_expand, 1, pos_topk_idxs)
        
    # 对每个类别的前k个样本计算平均预测概率，作为该类别的先验概率，args.rho是一个超参数，用于控制先验概率的平滑程度
    pos_cls_prior = torch.mean(sorted_pred_cls[:(pos_topk_num), :], dim=0, keepdim=True).t() * (1.0 - args.rho) + args.rho
    # 记录每个类别的先验概率
    args.logger.info("POS_CLS_PRIOR:\t" + "\t".join(["{:.3f}".format(item) for item in pos_cls_prior.cpu().squeeze().numpy()]))
    # 计算正原型                                                 
    pos_feat_proto = torch.mean(pos_feat_sample, dim=1, keepdim=True) #[C, 1, D]
    pos_feat_proto = pos_feat_proto / torch.norm(pos_feat_proto, p=2, dim=-1, keepdim=True)

    # 使用faiss库中的kmeans函数进行聚类
    faiss_kmeans = faiss.Kmeans(args.embed_feat_dim, KK, niter=100, verbose=False, min_points_per_centroid=1, gpu=False)
    

    feat_proto_pos_simi = torch.zeros((data_num, args.class_num)).cuda() #[N, C]
    feat_proto_max_simi = torch.zeros((data_num, args.class_num)).cuda() #[N, C]
    feat_proto_max_idxs = torch.zeros((data_num, args.class_num)).cuda() #[N, C]
    
    # One-vs-all class pseudo-labeling
    for cls_idx in range(args.class_num):
        # 对负样本聚类并计算原型
        neg_feat_cls_sample_np = torch.gather(embed_feat_bank, 0, neg_topk_idxs[cls_idx, :]).cpu().numpy()
        faiss_kmeans.train(neg_feat_cls_sample_np)
        cls_neg_feat_proto = torch.from_numpy(faiss_kmeans.centroids).cuda()
        cls_neg_feat_proto = cls_neg_feat_proto / torch.norm(cls_neg_feat_proto, p=2, dim=-1, keepdim=True)#[K, D]
        cls_pos_feat_proto = pos_feat_proto[cls_idx, :] #[1, D]
        
        # 计算正负原型和其他样本的相似度
        cls_pos_feat_proto_simi = torch.einsum("nd, kd -> nk", embed_feat_bank, cls_pos_feat_proto) #[N, 1]
        cls_neg_feat_proto_simi = torch.einsum("nd, kd -> nk", embed_feat_bank, cls_neg_feat_proto) #[N, K]
        cls_pos_feat_proto_simi = cls_pos_feat_proto_simi * pos_cls_prior[cls_idx] #[N, 1]
        
        cls_feat_proto_simi = torch.cat([cls_pos_feat_proto_simi, cls_neg_feat_proto_simi], dim=1) #[N, 1+K]
        
        # 样本与正原型的相似度
        feat_proto_pos_simi[:, cls_idx] = cls_feat_proto_simi[:, 0]
        # 最大相似度及对应索引
        maxsimi, maxidxs = torch.max(cls_feat_proto_simi, dim=-1)
        feat_proto_max_simi[:, cls_idx] = maxsimi
        feat_proto_max_idxs[:, cls_idx] = maxidxs
    
    # we use this psd_label_prior_simi to control the hard pseudo label either one-hot or unifrom distribution. 
    # 与正原型的相似度矩阵
    psd_label_prior_simi = torch.einsum("nd, cd -> nc", embed_feat_bank, pos_feat_proto.squeeze(1))
    psd_label_prior_idxs = torch.max(psd_label_prior_simi, dim=-1, keepdim=True)[1] #[N] ~ (0, class_num-1)
    # onehot编码的伪标签
    psd_label_prior = torch.zeros_like(psd_label_prior_simi).scatter(1, psd_label_prior_idxs, 1.0) # one_hot prior #[N, C]
    
    hard_psd_label_bank = feat_proto_max_idxs # [N, C] ~ (0, K)
    hard_psd_label_bank = (hard_psd_label_bank == 0).float()
    hard_psd_label_bank = hard_psd_label_bank * psd_label_prior #[N, C]
    
    # 最大概率对应类别作为硬标签
    hard_label = torch.argmax(hard_psd_label_bank, dim=-1) #[N]
    # 计算所有类别概率，若为0，则说明不属于任何类别，为未知类
    hard_label_unk = torch.sum(hard_psd_label_bank, dim=-1) 
    hard_label_unk = (hard_label_unk == 0)
    # 将未知类标签设为class_num，类别数加1，未知类被作为单独的类别处理
    hard_label[hard_label_unk] = args.class_num
    
    # 将未知类加入标签集，再归一化
    hard_psd_label_bank[hard_label_unk, :] += 1.0
    hard_psd_label_bank = hard_psd_label_bank / (torch.sum(hard_psd_label_bank, dim=-1, keepdim=True) + 1e-4)
    
    hard_psd_label_bank = hard_psd_label_bank.cuda()
    
    per_class_num = np.zeros((len(class_list)))
    pre_class_num = np.zeros_like(per_class_num)
    per_class_correct = np.zeros_like(per_class_num)
    # 计算准确率
    for i, label in enumerate(class_list):
        label_idx = torch.where(gt_label_bank == label)[0]
        correct_idx = torch.where(hard_label[label_idx] == label)[0]
        pre_class_num[i] = float(len(torch.where(hard_label == label)[0]))
        per_class_num[i] = float(len(label_idx))
        per_class_correct[i] = float(len(correct_idx))
    per_class_acc = per_class_correct / (per_class_num + 1e-5)
    
    # args.logger.info("PSD AVG ACC:\t" + "{:.3f}".format(np.mean(per_class_acc)))
    # args.logger.info("PSD PER ACC:\t" + "\t".join(["{:.3f}".format(item) for item in per_class_acc]))
    # args.logger.info("PER CLS NUM:\t" + "\t".join(["{:.0f}".format(item) for item in per_class_num]))
    # args.logger.info("PRE CLS NUM:\t" + "\t".join(["{:.0f}".format(item) for item in pre_class_num]))
    # args.logger.info("PRE ACC NUM:\t" + "\t".join(["{:.0f}".format(item) for item in per_class_correct]))
    
    return hard_psd_label_bank, pred_cls_bank, embed_feat_bank, best_coeff


def train(args, model, data_prompt, train_dataloader, test_dataloader, optimizer, epoch_idx=0.0, pred_unc_all=None):
    
    model.eval()
    # prompter.eval()
    hard_psd_label_bank, pred_cls_bank, embed_feat_bank, best_coeff = obtain_global_pseudo_labels(args, model, data_prompt, test_dataloader, epoch_idx)
    model.train()
    # dset_loaders, dsets = data_load(args)

    
    local_KNN = args.local_K
    all_pred_loss_stack = []
    psd_pred_loss_stack = []
    knn_pred_loss_stack = []
    unc_pred_loss_stack = []
    # im_loss_stack = []


    iter_idx = epoch_idx * len(train_dataloader)
    iter_max = args.epochs * len(train_dataloader)

    # index_bank = []
    # near_bank = []
    
    for imgs_train, _, _, imgs_idx in tqdm(train_dataloader, ncols=60):
        
        iter_idx += 1
        imgs_idx = imgs_idx.cuda()
        imgs_train = imgs_train.cuda()
        data_prompt = data_prompt.cuda()

        # index_bank.append(imgs_idx)
        # index_bank = torch.cat(index_bank, dim=0)  # [N, C]

        # imgs_train = mix_data_prompt(imgs_train, data_prompt)

        # inputs_train_stg = get_strong_aug(dsets["strong_aug"], imgs_idx)
        # inputs_train_stg = inputs_train_stg.cuda()
        # inputs_train_wk = imgs_train.cuda()
        # imgs_train = inputs_train_stg.cuda()
        # prompted_images = prompter(imgs_train)
        # embed_feat, pred_cls = model(imgs_train, apply_softmax=True)
        embed_feat, pred_cls = model.forward_pls(imgs_train, data_prompt, apply_softmax=True)
        # print (inputs_test_wk.shape)
        # print (inputs_test_stg.shape)
        # imgs_train = torch.cat([inputs_test_wk, imgs_train], dim=0)
        
        hard_psd_label = hard_psd_label_bank[imgs_idx] #[B, C]


        # 全局损失L(glb tar)
        psd_pred_loss = torch.sum(-hard_psd_label * torch.log(pred_cls + 1e-5), dim=-1).mean()  # type: Any
        # psd_pred_loss += torch.sum(-hard_psd_label * torch.log(pred_cls_stg + 1e-5), dim=-1).mean()  # type: Any
        # im_loss = torch.sqrt(embed_feat_stg - embed_feat)
        # im_loss = torch.ones_like(im_loss)

        # if iter_idx > 3:
        lam_psd = (1 + 10 * iter_idx / iter_max) ** (-args.beta) * args.lam_psd  #学习率衰减函数的权重衰减速率会随着迭代次数的增加而减小
        # else:
        #     lam_psd = args.lam_psd
        
        with torch.no_grad():
            # 将嵌入特征除以其L2范数，以便在计算余弦相似度时进行标准化
            embed_feat = embed_feat / torch.norm(embed_feat, p=2, dim=-1, keepdim=True)
            # 余弦相似度
            feat_dist = torch.einsum("bd, nd -> bn", embed_feat, embed_feat_bank) #[B, N]
            # 使用torch.topk选择余弦相似度值最高的k个元素
            nn_feat_idx = torch.topk(feat_dist, k=local_KNN+1, dim=-1, largest=True)[-1] #[B, local_KNN+1]
            nn_feat_idx = nn_feat_idx[:, 1:] #[B, local_KNN]

            # update the pred_cls and embed_feat bank 将当前图像的预测类别和嵌入特征存储在存储库中，以便在以后的迭代中使用
            pred_cls_bank[imgs_idx] = pred_cls
            embed_feat_bank[imgs_idx] = embed_feat
            # 计算这些特征的平均预测类别，以获得当前图像的预测类别
            nn_pred_cls = torch.mean(pred_cls_bank[nn_feat_idx], dim=1)  # [B, C]

            #NRC
            # distance = embed_feat @ embed_feat_bank.T
            # _, idx_near = torch.topk(distance,
            #                          dim=-1,
            #                          largest=True,
            #                          k=args.K+1)
            # idx_near = idx_near[:, 1:]  # batch x K
            # 余弦相似度值最高的k个元素的softmax值和特征值
            score_near = pred_cls_bank[nn_feat_idx]  # batch x K x C
            fea_near = embed_feat_bank[nn_feat_idx]  # batch x K x num_dim
            # fea_bank_re具有shape (batch_size, n, num_dim)，其中n是最近邻的最近邻的数量，num_dim是特征空间的维数
            fea_bank_re = embed_feat_bank.unsqueeze(0).expand(fea_near.shape[0], -1, -1)  # batch x n x dim
            distance_ = torch.bmm(fea_near, fea_bank_re.permute(0, 2, 1))  # batch x K x n
            _, idx_near_near = torch.topk(distance_, dim=-1, largest=True,
                                          k=args.KK+1)  # M near neighbors for each of above K ones， 形状为(batch_size, K, K+1)
            idx_near_near = idx_near_near[:, :, 1:]  # batch x K x M

            tar_idx_ = imgs_idx.unsqueeze(-1).unsqueeze(-1)
            match = (
                    idx_near_near == tar_idx_).sum(-1).float()  # batch x K
            weight = torch.where(
                match > 0., match,
                torch.ones_like(match).fill_(0.1))  # batch x K

            weight_kk = weight.unsqueeze(-1).expand(-1, -1,
                                                    args.KK)  # batch x K x M
            weight_kk = weight_kk.fill_(0.1)

            # removing the self in expanded neighbors, or otherwise you can keep it and not use extra self regularization
            # weight_kk[idx_near_near == tar_idx_]=0

            score_near_kk = pred_cls_bank[idx_near_near]  # batch x K x M x C
            # print(weight_kk.shape)
            weight_kk = weight_kk.contiguous().view(weight_kk.shape[0],
                                                    -1)  # batch x KM

            score_near_kk = score_near_kk.contiguous().view(score_near_kk.shape[0], -1,
                                                            args.class_num)  # batch x KM x C

            score_self = pred_cls_bank[imgs_idx]
            # nn_pred_cls = torch.mean(score_near_kk, dim=1)  # [B, C]
            #


        #
        #     # 熵最小化
        entropy_avg = torch.mean(Entropy(pred_cls))
        #     msoftmax = pred_cls.mean(dim=0)
        #     # 平均熵最大
        #     gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
        #     entropy_loss -= gentropy_loss
        # im_loss = entropy_loss #* args.ent_par

        # unknown_loss
        if pred_unc_all is None:
            pred_unc_all = Entropy(pred_cls_bank) / np.log(args.class_num)  # [N]
        unc_idx = torch.where(pred_unc_all > 0.55)[0]
        pred_unc = pred_cls_bank[unc_idx]

        unc_pred_loss = torch.mean(Entropy(pred_unc)) - entropy_avg
        # unc_pred_loss = 0.5 * torch.sqrt(unc_pred_loss * unc_pred_loss) - 0.5 * torch.mean(Entropy(pred_unc))
        unc_pred_loss = torch.sqrt(unc_pred_loss * unc_pred_loss)
        # unc_pred_loss = -torch.mean(Entropy(pred_unc))

        knn_pred_loss = torch.sum(-nn_pred_cls * torch.log(pred_cls + 1e-5), dim=-1).mean()

        # NRC
        # nn of nn
        output_re = pred_cls.unsqueeze(1).expand(-1, args.K * args.KK, -1)  # batch x C x 1
        const = torch.mean(
            (F.kl_div(output_re, score_near_kk, reduction='none').sum(-1) *
             weight_kk.cuda()).sum(1))  # kl_div here equals to dot product since we do not use log for score_near_kk
        knn_pred_loss += torch.mean(const)

        # nn
        softmax_out_un = pred_cls.unsqueeze(1).expand(-1, args.K, -1)  # batch x K x C

        knn_pred_loss += torch.mean((F.kl_div(softmax_out_un, score_near, reduction='none').sum(-1) *
                                   weight.cuda()).sum(1))

        # self, if not explicitly removing the self feature in expanded neighbor then no need for this
        knn_pred_loss += -torch.mean((pred_cls * score_self).sum(-1))
        #
        # msoftmax = pred_cls.mean(dim=0)
        # gentropy_loss = torch.sum(msoftmax * torch.log(msoftmax + args.epsilon))
        # knn_pred_loss += gentropy_loss



        # 总损失，psd为全局损失，knn为局部损失
        # if best_coeff > 1 :
        # loss = knn_pred_loss
        loss = lam_psd * psd_pred_loss + (1 - lam_psd) * knn_pred_loss #- 0.1 * unc_pred_loss #+ args.lam_im * im_loss
        # else:
        #     loss = (1 - lam_knn) * psd_pred_loss + lam_knn * knn_pred_loss
        
        lr_scheduler(optimizer, iter_idx, iter_max)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        all_pred_loss_stack.append(loss.cpu().item())
        psd_pred_loss_stack.append(psd_pred_loss.cpu().item())
        knn_pred_loss_stack.append(knn_pred_loss.cpu().item())
        unc_pred_loss_stack.append(unc_pred_loss.cpu().item())
        # im_loss_stack.append(im_loss.cpu().item())
        
    train_loss_dict = {}
    train_loss_dict["all_pred_loss"] = np.mean(all_pred_loss_stack)
    train_loss_dict["psd_pred_loss"] = np.mean(psd_pred_loss_stack)
    train_loss_dict["knn_pred_loss"] = np.mean(knn_pred_loss_stack)
    train_loss_dict["unc_pred_loss"] = np.mean(unc_pred_loss_stack)
    # train_loss_dict["im_loss"] = np.mean(im_loss_stack)
            
    return train_loss_dict, pred_unc_all
    
@torch.no_grad()
def test(args, model, data_prompt, dataloader, src_flg=False):
    
    model.eval()
    gt_label_stack = []
    embed_feat_stack = []
    pred_cls_stack = []
    
    if src_flg:
        class_list = args.source_class_list
        open_flg = False
    else:
        class_list = args.target_class_list
        open_flg = args.target_private_class_num > 0
    
    for _, imgs_test, imgs_label, _ in tqdm(dataloader, ncols=60):
        
        imgs_test = imgs_test.cuda()
        # prompted_images = prompter(imgs_test)
        _, pred_cls = model(imgs_test, apply_softmax=True)
        embed_feat, pred_cls = model.forward_pls(imgs_test, data_prompt, apply_softmax=True)
        gt_label_stack.append(imgs_label)
        embed_feat_stack.append(embed_feat)
        pred_cls_stack.append(pred_cls.cpu())
    
    gt_label_all = torch.cat(gt_label_stack, dim=0) #[N]
    label_all = torch.cat(gt_label_stack, axis=0).cpu().numpy()  # [N]
    embed_feat_all = torch.cat(embed_feat_stack, axis=0).cpu().numpy()
    pred_cls_all = torch.cat(pred_cls_stack, dim=0) #[N, C]

    np.save("embed_feat_Cl2Re.npy", embed_feat_all)
    np.save("label_Cl2Re.npy", label_all)

    h_score, known_acc, unknown_acc, _ = compute_h_score(args, class_list, gt_label_all, pred_cls_all, open_flg, open_thresh=args.w_0)
    return h_score, known_acc, unknown_acc

# def visualization(args, features):
#     vec_list = features.cpu().numpy()  # features是要聚类的特征
#     vec_array = np.array(vec_list)
#     tsne = manifold.TSNE(n_components=2, init='pca', random_state=42, perplexity=8, learning_rate=515,
#                          n_iter=50000).fit_transform(vec_array)
#     # tsne 归一化
#     x_min, x_max = tsne.min(0), tsne.max(0)
#     tsne_norm = (tsne - x_min) / (x_max - x_min)
#     # 每类取50个点
#
#
#     tsne_1 = tsne_norm[0:50]
#     tsne_2 = tsne_norm[50:100]
#     tsne_3 = tsne_norm[100:150]
#     tsne_4 = tsne_norm[150:200]
#     plt.figure(figsize=(8, 8))
#     plt.scatter(tsne_1[:, 0], tsne_1[:, 1], 30, label='Brown Creeper')
#     # tsne_normal[i, 0]为横坐标，X_norm[i, 1]为纵坐标，1为散点图的面积， color给每个类别设定颜色
#     plt.scatter(tsne_2[:, 0], tsne_2[:, 1], 30, label='Rusty Blackbird')
#     plt.scatter(tsne_3[:, 0], tsne_3[:, 1], 30, label='Horned Grebe')
#     plt.scatter(tsne_4[:, 0], tsne_4[:, 1], 30, label='Ovenbird')
#     plt.legend(loc="lower right")
#     plt.show()
    
def main(args):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    this_dir = os.path.join(os.path.dirname(__file__), ".")
    
    model = SFUniDA(args)
    
    if args.checkpoint is not None and os.path.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=torch.device("cuda"))
        model.load_state_dict(checkpoint["model_state_dict"], False)
        # data_prompt = checkpoint["model_state_dict"]["data_prompt"]
    else:
        print(args.checkpoint)
        raise ValueError("YOU MUST SET THE APPROPORATE SOURCE CHECKPOINT FOR TARGET MODEL ADPTATION!!!")


    prompt_model_params = torch.load(args.prompt_model_path, map_location=torch.device("cuda"))
    data_prompt = prompt_model_params['data_prompt'].cuda().to(dtype=torch.float32)
    # print(data_prompt)
    
    model = model.cuda()
    save_dir = os.path.join(this_dir, "checkpoints_glc", args.dataset, "s_{}_to_t_{}".format(args.s_idx, args.t_idx),
                            args.target_label_type, args.note)
    
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    args.save_dir = save_dir
    args.logger = set_logger(args, log_name="log_target_training.txt")
    
    param_group = []
    for k, v in model.backbone_layer.named_parameters():
        param_group += [{'params': v, 'lr': args.lr*0.1}]
    
    for k, v in model.feat_embed_layer.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]
    
    for k, v in model.class_layer.named_parameters():
        v.requires_grad = False

    # pretrained_params = checkpoint

    optimizer = torch.optim.SGD(param_group)
    optimizer = op_copy(optimizer)
    
    target_data_list = open(os.path.join(args.target_data_dir, "image_unida_list.txt"), "r").readlines()
    target_dataset = SFUniDADataset(args, args.target_data_dir, target_data_list, d_type="target", preload_flg=True)
    
    target_train_dataloader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True,
                                         num_workers=args.num_workers, drop_last=True)
    target_test_dataloader = DataLoader(target_dataset, batch_size=args.batch_size*2, shuffle=False,
                                        num_workers=args.num_workers, drop_last=False)
    
    notation_str =  "\n=======================================================\n"
    notation_str += "   START TRAINING ON THE TARGET:{} BASED ON SOURCE:{}  \n".format(args.t_idx, args.s_idx)
    notation_str += "======================================================="
    
    args.logger.info(notation_str)
    best_h_score = 0.0
    best_known_acc = 0.0
    best_unknown_acc = 0.0
    best_epoch_idx = 0
    for epoch_idx in tqdm(range(args.epochs), ncols=60):
        # loss_dict, data_prompt = train_prompt(args, model, pretrained_params, target_train_dataloader, target_test_dataloader,
        #                                optimizer, epoch_idx)
        # args.logger.info("Epoch: {}/{},          train_all_loss:{:.3f}".format(epoch_idx + 1, args.epochs,
        #                                                                        loss_dict["loss"]))
        # Train on target
        loss_dict, pred_unc_all =train(args, model, data_prompt, target_train_dataloader, target_test_dataloader, optimizer, epoch_idx)
        args.logger.info("Epoch: {}/{},          train_all_loss:{:.3f},\n\
                          train_psd_loss:{:.3f}, train_knn_loss:{:.3f}, train_unc_loss:{:.3f}".format(epoch_idx+1, args.epochs,
                                        loss_dict["all_pred_loss"], loss_dict["psd_pred_loss"], loss_dict["knn_pred_loss"], loss_dict["unc_pred_loss"]))
        # print (pred_unc_all)
        # args.logger.info("Epoch: {}/{},          train_all_loss:{:.3f},\n\
        #                           train_psd_loss:{:.3f}, train_knn_loss:{:.3f}".format(epoch_idx + 1, args.epochs,
        #                             loss_dict["all_pred_loss"], loss_dict["psd_pred_loss"], loss_dict["knn_pred_loss"]))
        # Evaluate on target
        hscore, knownacc, unknownacc = test(args, model, data_prompt, target_test_dataloader, src_flg=False)
        args.logger.info("Current: H-Score:{:.3f}, KnownAcc:{:.3f}, UnknownAcc:{:.3f}".format(hscore, knownacc, unknownacc))
        
        if args.target_label_type == 'PDA' or args.target_label_type == 'CLDA':
            if knownacc >= best_known_acc:
                best_h_score = hscore
                best_known_acc = knownacc
                best_unknown_acc = unknownacc
                best_epoch_idx = epoch_idx

                # checkpoint_file = "{}_SFDA_best_target_checkpoint.pth".format(args.dataset)
                # torch.save({
                #     "epoch":epoch_idx,
                #     "model_state_dict":model.state_dict()}, os.path.join(save_dir, checkpoint_file))
        else:
            if hscore >= best_h_score:
                best_h_score = hscore
                best_known_acc = knownacc
                best_unknown_acc = unknownacc
                best_epoch_idx = epoch_idx
            
                checkpoint_file = "{}_SFDA_best_target_checkpoint.pth".format(args.dataset)
                torch.save({
                    "epoch":epoch_idx,
                    "model_state_dict":model.state_dict()}, os.path.join(save_dir, checkpoint_file))
            
        args.logger.info("Best   : H-Score:{:.3f}, KnownAcc:{:.3f}, UnknownAcc:{:.3f}".format(best_h_score, best_known_acc, best_unknown_acc))
            
if __name__ == "__main__":
    args = build_args()
    set_random_seed(args.seed)
    
    # SET THE CHECKPOINT     
    args.checkpoint = os.path.join("checkpoints_glc", args.dataset, "s_{}_t_{}".format(args.s_idx, args.t_idx),
                            args.target_label_type, args.note, "latest_source_checkpoint.pth")
    # args.checkpoint = os.path.join("checkpoints_glc", args.dataset, "source_{}".format(args.s_idx), \
    #                                "source_{}_{}".format(args.source_train_type, args.target_label_type),
    #                                "latest_source_checkpoint.pth")
    args.prompt_model_path = os.path.join("checkpoints_glc", args.dataset, "s_{}_t_{}".format(args.s_idx, args.t_idx),
                            args.target_label_type, args.note, "prompt_model.pth")
    main(args)
