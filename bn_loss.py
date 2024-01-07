# -*- coding:utf-8 -*-
import torch.nn.functional as F
import torch
from config.model_config import build_args


def calculate_one_layer(this, stored, alpha=0.01):
    return (this[0] - stored[0]).abs().mean() + (this[1] - stored[1]).abs().mean() * alpha
    # return ((this[0] - stored[0])*(this[0] - stored[0])).mean() + ((this[1] - stored[1])*(this[1] - stored[1])).mean() * alpha


def layer_1_loss(model, pretrained_params, bn_f, alpha=0.01):
    # pretrained_params = pretrained_params.to(args.gpu)
    # bn_f = bn_f.to(args.gpu)
    # pretrained_params.device = 'cuda'
    # print (bn_f.device)
    loss = calculate_one_layer([bn_f[0].features.mean(dim=(0)), bn_f[0].features.var(dim=(0))],
                                     [pretrained_params['model_state_dict']['feat_embed_layer.bn.running_mean'],
                                      pretrained_params['model_state_dict']['feat_embed_layer.bn.running_var']], alpha=alpha)
    return loss


def layer_2_loss(model, pretrained_params, bn_f, alpha=0.01):
    loss = calculate_one_layer([bn_f[2].features.mean(dim=(0, -2, -1)), bn_f[2].features.var(dim=(0, -2, -1))],
                               [pretrained_params['model_state_dict']['backbone_layer.layer1.0.bn1.running_mean'],
                                pretrained_params['model_state_dict']['backbone_layer.layer1.0.bn1.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[3].features.mean(dim=(0, -2, -1)), bn_f[3].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['backbone_layer.layer1.0.bn2.running_mean'],
                                      pretrained_params['model_state_dict']['backbone_layer.layer1.0.bn2.running_var']], alpha=alpha)  + \
                 calculate_one_layer([bn_f[4].features.mean(dim=(0, -2, -1)), bn_f[4].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['backbone_layer.layer1.0.bn3.running_mean'],
                                      pretrained_params['model_state_dict']['backbone_layer.layer1.0.bn3.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[5].features.mean(dim=(0, -2, -1)), bn_f[5].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['backbone_layer.layer1.1.bn1.running_mean'],
                                      pretrained_params['model_state_dict']['backbone_layer.layer1.1.bn1.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[6].features.mean(dim=(0, -2, -1)), bn_f[6].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['backbone_layer.layer1.1.bn2.running_mean'],
                                      pretrained_params['model_state_dict']['backbone_layer.layer1.1.bn2.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[7].features.mean(dim=(0, -2, -1)), bn_f[7].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['backbone_layer.layer1.1.bn3.running_mean'],
                                      pretrained_params['model_state_dict']['backbone_layer.layer1.1.bn3.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[8].features.mean(dim=(0, -2, -1)), bn_f[8].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['backbone_layer.layer1.2.bn1.running_mean'],
                                      pretrained_params['model_state_dict']['backbone_layer.layer1.2.bn1.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[9].features.mean(dim=(0, -2, -1)), bn_f[9].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['backbone_layer.layer1.2.bn2.running_mean'],
                                      pretrained_params['model_state_dict']['backbone_layer.layer1.2.bn2.running_var']], alpha=alpha)  + \
                calculate_one_layer([bn_f[10].features.mean(dim=(0, -2, -1)), bn_f[10].features.var(dim=(0, -2, -1))],
                                    [pretrained_params['model_state_dict']['backbone_layer.layer1.2.bn3.running_mean'],
                                    pretrained_params['model_state_dict']['backbone_layer.layer1.2.bn3.running_var']], alpha=alpha)
    return loss


def layer_3_loss(model, pretrained_params, bn_f, alpha=0.01):
    loss = calculate_one_layer([bn_f[11].features.mean(dim=(0, -2, -1)), bn_f[11].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['backbone_layer.layer2.0.bn1.running_mean'],
                                      pretrained_params['model_state_dict']['backbone_layer.layer2.0.bn1.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[12].features.mean(dim=(0, -2, -1)), bn_f[12].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['backbone_layer.layer2.0.bn2.running_mean'],
                                      pretrained_params['model_state_dict']['backbone_layer.layer2.0.bn2.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[13].features.mean(dim=(0, -2, -1)), bn_f[13].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['backbone_layer.layer2.0.bn3.running_mean'],
                                      pretrained_params['model_state_dict']['backbone_layer.layer2.0.bn3.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[14].features.mean(dim=(0, -2, -1)), bn_f[14].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['backbone_layer.layer2.1.bn1.running_mean'],
                                      pretrained_params['model_state_dict']['backbone_layer.layer2.1.bn1.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[15].features.mean(dim=(0, -2, -1)), bn_f[15].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['backbone_layer.layer2.1.bn2.running_mean'],
                                      pretrained_params['model_state_dict']['backbone_layer.layer2.1.bn2.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[16].features.mean(dim=(0, -2, -1)), bn_f[16].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['backbone_layer.layer2.1.bn3.running_mean'],
                                      pretrained_params['model_state_dict']['backbone_layer.layer2.1.bn3.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[17].features.mean(dim=(0, -2, -1)), bn_f[17].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['backbone_layer.layer2.2.bn1.running_mean'],
                                      pretrained_params['model_state_dict']['backbone_layer.layer2.2.bn1.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[18].features.mean(dim=(0, -2, -1)), bn_f[18].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['backbone_layer.layer2.2.bn2.running_mean'],
                                      pretrained_params['model_state_dict']['backbone_layer.layer2.2.bn2.running_var']], alpha=alpha)  + \
                 calculate_one_layer([bn_f[19].features.mean(dim=(0, -2, -1)), bn_f[19].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['backbone_layer.layer2.2.bn3.running_mean'],
                                      pretrained_params['model_state_dict']['backbone_layer.layer2.2.bn3.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[20].features.mean(dim=(0, -2, -1)), bn_f[20].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['backbone_layer.layer2.3.bn1.running_mean'],
                                     pretrained_params['model_state_dict']['backbone_layer.layer2.3.bn1.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[21].features.mean(dim=(0, -2, -1)), bn_f[21].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['backbone_layer.layer2.3.bn2.running_mean'],
                                      pretrained_params['model_state_dict']['backbone_layer.layer2.3.bn2.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[22].features.mean(dim=(0, -2, -1)), bn_f[22].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['backbone_layer.layer2.3.bn3.running_mean'],
                                      pretrained_params['model_state_dict']['backbone_layer.layer2.3.bn3.running_var']], alpha=alpha)
    return loss


def layer_4_loss(model, pretrained_params, bn_f, alpha=0.01):
    loss = calculate_one_layer([bn_f[23].features.mean(dim=(0, -2, -1)), bn_f[23].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['backbone_layer.layer3.0.bn1.running_mean'],
                                      pretrained_params['model_state_dict']['backbone_layer.layer3.0.bn1.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[24].features.mean(dim=(0, -2, -1)), bn_f[24].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['backbone_layer.layer3.0.bn2.running_mean'],
                                      pretrained_params['model_state_dict']['backbone_layer.layer3.0.bn2.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[25].features.mean(dim=(0, -2, -1)), bn_f[25].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['backbone_layer.layer3.0.bn3.running_mean'],
                                      pretrained_params['model_state_dict']['backbone_layer.layer3.0.bn3.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[26].features.mean(dim=(0, -2, -1)), bn_f[26].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['backbone_layer.layer3.1.bn1.running_mean'],
                                      pretrained_params['model_state_dict']['backbone_layer.layer3.1.bn1.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[27].features.mean(dim=(0, -2, -1)), bn_f[27].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['backbone_layer.layer3.1.bn2.running_mean'],
                                      pretrained_params['model_state_dict']['backbone_layer.layer3.1.bn2.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[28].features.mean(dim=(0, -2, -1)), bn_f[28].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['backbone_layer.layer3.1.bn3.running_mean'],
                                      pretrained_params['model_state_dict']['backbone_layer.layer3.1.bn3.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[29].features.mean(dim=(0, -2, -1)), bn_f[29].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['backbone_layer.layer3.2.bn1.running_mean'],
                                      pretrained_params['model_state_dict']['backbone_layer.layer3.2.bn1.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[30].features.mean(dim=(0, -2, -1)), bn_f[30].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['backbone_layer.layer3.2.bn2.running_mean'],
                                      pretrained_params['model_state_dict']['backbone_layer.layer3.2.bn2.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[31].features.mean(dim=(0, -2, -1)), bn_f[31].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['backbone_layer.layer3.2.bn3.running_mean'],
                                      pretrained_params['model_state_dict']['backbone_layer.layer3.2.bn3.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[32].features.mean(dim=(0, -2, -1)), bn_f[32].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['backbone_layer.layer3.3.bn1.running_mean'],
                                      pretrained_params['model_state_dict']['backbone_layer.layer3.3.bn1.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[33].features.mean(dim=(0, -2, -1)), bn_f[33].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['backbone_layer.layer3.3.bn2.running_mean'],
                                      pretrained_params['model_state_dict']['backbone_layer.layer3.3.bn2.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[34].features.mean(dim=(0, -2, -1)), bn_f[34].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['backbone_layer.layer3.3.bn3.running_mean'],
                                      pretrained_params['model_state_dict']['backbone_layer.layer3.3.bn3.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[35].features.mean(dim=(0, -2, -1)), bn_f[35].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['backbone_layer.layer3.4.bn1.running_mean'],
                                      pretrained_params['model_state_dict']['backbone_layer.layer3.4.bn1.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[36].features.mean(dim=(0, -2, -1)), bn_f[36].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['backbone_layer.layer3.4.bn2.running_mean'],
                                      pretrained_params['model_state_dict']['backbone_layer.layer3.4.bn2.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[37].features.mean(dim=(0, -2, -1)), bn_f[37].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['backbone_layer.layer3.4.bn3.running_mean'],
                                      pretrained_params['model_state_dict']['backbone_layer.layer3.4.bn3.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[38].features.mean(dim=(0, -2, -1)), bn_f[38].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['backbone_layer.layer3.5.bn1.running_mean'],
                                      pretrained_params['model_state_dict']['backbone_layer.layer3.5.bn1.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[39].features.mean(dim=(0, -2, -1)), bn_f[39].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['backbone_layer.layer3.5.bn2.running_mean'],
                                      pretrained_params['model_state_dict']['backbone_layer.layer3.5.bn2.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[40].features.mean(dim=(0, -2, -1)), bn_f[40].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['backbone_layer.layer3.5.bn3.running_mean'],
                                      pretrained_params['model_state_dict']['backbone_layer.layer3.5.bn3.running_var']], alpha=alpha)
    return loss


def layer_5_loss(model, pretrained_params, bn_f, alpha=0.01):
    loss = calculate_one_layer([bn_f[41].features.mean(dim=(0, -2, -1)), bn_f[41].features.var(dim=(0, -2, -1))],
                               [pretrained_params['model_state_dict']['backbone_layer.layer4.0.bn1.running_mean'],
                                pretrained_params['model_state_dict']['backbone_layer.layer4.0.bn1.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[42].features.mean(dim=(0, -2, -1)), bn_f[42].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['backbone_layer.layer4.0.bn2.running_mean'],
                                      pretrained_params['model_state_dict']['backbone_layer.layer4.0.bn2.running_var']], alpha=alpha)  + \
                 calculate_one_layer([bn_f[43].features.mean(dim=(0, -2, -1)), bn_f[43].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['backbone_layer.layer4.0.bn3.running_mean'],
                                      pretrained_params['model_state_dict']['backbone_layer.layer4.0.bn3.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[44].features.mean(dim=(0, -2, -1)), bn_f[44].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['backbone_layer.layer4.1.bn1.running_mean'],
                                      pretrained_params['model_state_dict']['backbone_layer.layer4.1.bn1.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[45].features.mean(dim=(0, -2, -1)), bn_f[45].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['backbone_layer.layer4.1.bn2.running_mean'],
                                      pretrained_params['model_state_dict']['backbone_layer.layer4.1.bn2.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[46].features.mean(dim=(0, -2, -1)), bn_f[46].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['backbone_layer.layer4.1.bn3.running_mean'],
                                      pretrained_params['model_state_dict']['backbone_layer.layer4.1.bn3.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[47].features.mean(dim=(0, -2, -1)), bn_f[47].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['backbone_layer.layer4.2.bn1.running_mean'],
                                      pretrained_params['model_state_dict']['backbone_layer.layer4.2.bn1.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[48].features.mean(dim=(0, -2, -1)), bn_f[48].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['backbone_layer.layer4.2.bn2.running_mean'],
                                      pretrained_params['model_state_dict']['backbone_layer.layer4.2.bn2.running_var']], alpha=alpha)  + \
                calculate_one_layer([bn_f[49].features.mean(dim=(0, -2, -1)), bn_f[49].features.var(dim=(0, -2, -1))],
                                    [pretrained_params['model_state_dict']['backbone_layer.layer4.2.bn3.running_mean'],
                                     pretrained_params['model_state_dict']['backbone_layer.layer4.2.bn3.running_var']], alpha=alpha)
    return loss


def layer_6_loss(model, pretrained_params, bn_f, alpha=0.01):
    f = F.relu(torch.cat([bn_f[36].features, bn_f[37].features], dim=1))
    loss = calculate_one_layer([bn_f[1].features.mean(dim=(0, -2, -1)), bn_f[1].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['backbone_layer.bn1.running_mean'],
                                      pretrained_params['model_state_dict']['backbone_layer.bn1.running_var']], alpha=alpha) + \
           calculate_one_layer([f.mean(dim=(0, -2, -1)), f.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['up1.bn.running_mean'],
                                      pretrained_params['model_state_dict']['up1.bn.running_var']], alpha=alpha)
    return loss


def layer_7_loss(model, pretrained_params, bn_f, alpha=0.01):
    f = F.relu(torch.cat([bn_f[38].features, bn_f[39].features], dim=1))
    loss = calculate_one_layer([f.mean(dim=(0, -2, -1)), f.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['up2.bn.running_mean'],
                                      pretrained_params['model_state_dict']['up2.bn.running_var']], alpha=alpha)
    return loss


def layer_8_loss(model, pretrained_params, bn_f, alpha=0.01):
    f = F.relu(torch.cat([bn_f[40].features, bn_f[41].features], dim=1))
    loss = calculate_one_layer([f.mean(dim=(0, -2, -1)), f.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['up3.bn.running_mean'],
                                      pretrained_params['model_state_dict']['up3.bn.running_var']], alpha=alpha)
    return loss


def layer_9_loss(model, pretrained_params, bn_f, alpha=0.01):
    f = F.relu(torch.cat([bn_f[42].features, bn_f[43].features], dim=1))
    loss = calculate_one_layer([f.mean(dim=(0, -2, -1)), f.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['up4.bn.running_mean'],
                                      pretrained_params['model_state_dict']['up4.bn.running_var']], alpha=alpha)
    return loss


def bn_loss(model, pretrained_params, bn_f, alpha=0.01, i=1):
    loss_list = [
        layer_1_loss(model, pretrained_params, bn_f, alpha=alpha),
        layer_2_loss(model, pretrained_params, bn_f, alpha=alpha),
        layer_3_loss(model, pretrained_params, bn_f, alpha=alpha),
        layer_4_loss(model, pretrained_params, bn_f, alpha=alpha),
        layer_5_loss(model, pretrained_params, bn_f, alpha=alpha),
        # layer_6_loss(model, pretrained_params, bn_f, alpha=alpha),
        # layer_7_loss(model, pretrained_params, bn_f, alpha=alpha),
        # layer_8_loss(model, pretrained_params, bn_f, alpha=alpha),
        # layer_9_loss(model, pretrained_params, bn_f, alpha=alpha)
    ]

    total_loss = 0
    # for n in range(i):
    #     total_loss += loss_list[n]
    total_loss += loss_list[1] + loss_list[2] + loss_list[3] + loss_list[4]
    # total_loss += loss_list[1]
    return total_loss
