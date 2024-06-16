import torch 
import numpy as np 
import torch.nn as nn
from torchvision import models
import clip

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

vgg_dict = {"vgg11":models.vgg11, "vgg13":models.vgg13, 
            "vgg16":models.vgg16, "vgg19":models.vgg19, 
            "vgg11bn":models.vgg11_bn, "vgg13bn":models.vgg13_bn,
            "vgg16bn":models.vgg16_bn, "vgg19bn":models.vgg19_bn} 

class VGGBase(nn.Module):
  def __init__(self, vgg_name):
    super(VGGBase, self).__init__()
    model_vgg = vgg_dict[vgg_name](pretrained=True)
    self.features = model_vgg.features
    self.classifier = nn.Sequential()
    for i in range(6):
        self.classifier.add_module("classifier"+str(i), model_vgg.classifier[i])
    # self.in_features = model_vgg.classifier[6].in_features
    self.backbone_feat_dim = model_vgg.classifier[6].in_features
  
  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x

res_dict = {"resnet18":models.resnet18, "resnet34":models.resnet34,
            "resnet50":models.resnet50, "resnet101":models.resnet101,
            "resnet152":models.resnet152, "resnext50":models.resnext50_32x4d,
            "resnext101":models.resnext101_32x8d}

class ResBase(nn.Module):
    def __init__(self, res_name):
        super(ResBase, self).__init__()
        model_resnet = res_dict[res_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool

        self.bottleneck = nn.Linear(2048, 3136)
        # self.bottleneck1 = nn.Linear(2048, 2352)
        self.conv2 = nn.Conv2d(in_channels=65, out_channels=64,
                               kernel_size=1, stride=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64,
                               kernel_size=1, stride=1, bias=False)
        # self.conv3 = nn.Conv2d(in_channels=515, out_channels=512,
        #                        kernel_size=1, stride=1, bias=False)

        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.backbone_feat_dim = model_resnet.fc.in_features


    def forward(self, x):
        # stage0
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # stage1
        x = self.layer1(x)
        # stage2
        x = self.layer2(x)
        # stage3
        x = self.layer3(x)
        # stage4
        x = self.layer4(x)
        x = self.avgpool(x)  #(batch_size, channels, 1, 1)
        x = x.view(x.size(0), -1)
        return x

    def forward_with_semantic_prompt(self, x, semantic_prompt):
        # stage0
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        semantic_prompt = self.bottleneck(semantic_prompt)
        B, C, H, W = x.shape
        # print (semantic_prompt.shape)
        semantic_prompt = semantic_prompt.view(B, -1, H, W)
        # print (semantic_prompt.shape)
        # x_cat = x.view(B, -1, 1, 1)
        # print (x_cat.shape)
        x_cat = torch.cat([x, semantic_prompt], dim=1)
        # print (x_cat.shape)
        # x_cat = x_cat.view(B, -1, H, W)
        x = self.conv2(x_cat)
        x = self.conv3(x)
        # print (x.shape)
        # x = x.view(B, C, H, W)

        # stage1
        x = self.layer1(x)
        # stage2
        x = self.layer2(x)
        # semantic_prompt1 = self.bottleneck1(semantic_prompt)
        # B, C, H, W = x.shape
        # semantic_prompt1 = semantic_prompt1.view(B, -1, H, W)
        # x_cat = torch.cat([x, semantic_prompt1], dim=1)
        # x = self.conv3(x_cat)
        # stage3
        x = self.layer3(x)
        # stage4
        x = self.layer4(x)
        x = self.avgpool(x)  #(batch_size, channels, 1, 1)
        x = x.view(x.size(0), -1)
        # print (x.shape)
        return x


#
# class Conv(nn.Module):
#
#     def __init__(self, backbone_feat_dim):
#         super(Conv, self).__init__()
#         # self.global_pooling = nn.AdaptiveAvgPool2d(1)
#         self.bottleneck = nn.Linear(backbone_feat_dim * 2, backbone_feat_dim)
#         # self.bottleneck2 = nn.Linear(backbone_feat_dim, backbone_feat_dim)
#         # self.bottleneck1 = nn.Linear(backbone_feat_dim, 2048)
#         # self.bottleneck = nn.Conv2d(1, feature_dim, kernel_size=1)
#
#     def forward(self, x):
#         # x = self.global_pooling(x)
#         # x = x.unsqueeze(2).unsqueeze(3)
#         # x = x.transpose()
#         x = x.view(x.size(0), -1)
#         x = self.bottleneck(x)
#         # x = self.bottleneck2(x)
#         # x = self.bottleneck1(x)
#         # x = x.view(x.size(0), -1)
#         return x

# class Bottleneck1(nn.Module):
#     """
#     注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
#     但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
#     这么做的好处是能够在top1上提升大概0.5%的准确率。
#     可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
#     """
#     expansion = 4
#
#     def __init__(self, in_channel, out_channel, stride=1, downsample=None,
#                  groups=1, width_per_group=64):
#         super(Bottleneck1, self).__init__()
#
#         width = int(out_channel * (width_per_group / 64.)) * groups
#
#         self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
#                                kernel_size=1, stride=1, bias=False)  # squeeze channels
#         self.bn1 = nn.BatchNorm2d(width)
#         # -----------------------------------------
#         self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
#                                kernel_size=3, stride=stride, bias=False, padding=1)
#         self.bn2 = nn.BatchNorm2d(width)
#         # -----------------------------------------
#         self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
#                                kernel_size=1, stride=1, bias=False)  # unsqueeze channels
#         self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#
#     def forward(self, x):
#         identity = x
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         out += identity
#         out = self.relu(out)
#
#         return out
#
#
# class Bottleneck2(nn.Module):
#     """
#     注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
#     但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
#     这么做的好处是能够在top1上提升大概0.5%的准确率。
#     可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
#     """
#     expansion = 4
#
#     def __init__(self, in_channel, out_channel, stride=1, downsample=None,
#                  groups=1, width_per_group=64):
#         super(Bottleneck2, self).__init__()
#
#         width = int(out_channel * (width_per_group / 64.)) * groups
#
#         self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
#                                kernel_size=1, stride=1, bias=False)  # squeeze channels
#         self.bn1 = nn.BatchNorm2d(width)
#         # -----------------------------------------
#         self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
#                                kernel_size=3, stride=stride, bias=False, padding=1)
#         self.bn2 = nn.BatchNorm2d(width)
#         # -----------------------------------------
#         self.bottleneck = nn.Linear(width * 2, width)
#         self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
#                                kernel_size=1, stride=1, bias=False)  # unsqueeze channels
#         self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#
#     def forward(self, x, semantic_prompt):
#         identity = x
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         B, C, H, W = out.shape
#
#         semantic_prompt = semantic_prompt.view(B, C, 1, 1)
#         # semantic_prompt = self.Conv_layer(semantic_prompt)
#         backbone_feat = torch.cat([out, semantic_prompt], dim=-1)
#         # backbone_feat = backbone_feat.view(backbone_feat.size(0), -1)
#         # print (backbone_feat.shape)
#         out = self.bottleneck(backbone_feat)
#         # backbone_feat = torch.cat([backbone_feat1, backbone_feat], dim=-1)
#         # backbone_feat = self.Conv_layer(backbone_feat)
#         # backbone_feat = backbone_feat.view(backbone_feat.size(0), -1)  # [4,2048]
#         out = self.bottleneck(out)
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         out += identity
#         out = self.relu(out)
#
#         return out
#
# class ResNet(nn.Module):
#
#     def __init__(self,
#                  block1,
#                  blocks_num,
#                  num_classes=1000,
#                  include_top=True,
#                  groups=1,
#                  width_per_group=64):
#         super(ResNet, self).__init__()
#         self.include_top = include_top
#         self.in_channel = 64
#
#         self.groups = groups
#         self.width_per_group = width_per_group
#
#         self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
#                                padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(self.in_channel)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block1, 64, blocks_num[0])
#         self.layer2 = self._make_layer(block1, 128, blocks_num[1], stride=2)
#         self.layer3 = self._make_layer(block1, 256, blocks_num[2], stride=2)
#         self.layer4 = self._make_layer2(block1, 512, blocks_num[3], stride=2)
#         self.bottleneck = nn.Linear(2048, 1024)
#         # self.layer5 = Bottleneck2
#         # self.layer6 = Bottleneck2
#         # self.layer7 = Bottleneck2
#         if self.include_top:
#             self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
#             # self.fc = nn.Linear(512 * block1.expansion, num_classes)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#
#         self.backbone_feat_dim = 512 * block1.expansion
#
#     def _make_layer(self, block, channel, block_num, stride=1):
#         downsample = None
#         if stride != 1 or self.in_channel != channel * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(channel * block.expansion))
#
#         layers = []
#         layers.append(block(self.in_channel,
#                             channel,
#                             downsample=downsample,
#                             stride=stride,
#                             groups=self.groups,
#                             width_per_group=self.width_per_group))
#         self.in_channel = channel * block.expansion
#
#         for _ in range(1, block_num):
#             layers.append(block(self.in_channel,
#                                 channel,
#                                 groups=self.groups,
#                                 width_per_group=self.width_per_group))
#
#         return nn.Sequential(*layers)
#
#     def _make_layer2(self, block, channel, block_num, stride=1):
#         downsample = None
#         if stride != 1 or self.in_channel != channel * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(channel * block.expansion))
#
#         layers = []
#         layers.append(block(self.in_channel,
#                             channel,
#                             downsample=downsample,
#                             stride=stride,
#                             groups=self.groups,
#                             width_per_group=self.width_per_group))
#
#         return nn.Sequential(*layers)
#     #
#     # def _make_layer3(self, block, channel, block_num, stride=1):
#     #     # downsample = None
#     #     # if stride != 1 or self.in_channel != channel * block1.expansion:
#     #     #     downsample = nn.Sequential(
#     #     #         nn.Conv2d(self.in_channel, channel * block1.expansion, kernel_size=1, stride=stride, bias=False),
#     #     #         nn.BatchNorm2d(channel * block1.expansion))
#     #
#     #     layers = []
#     #     # layers.append(block1(self.in_channel,
#     #     #                     channel,
#     #     #                     downsample=downsample,
#     #     #                     stride=stride,
#     #     #                     groups=self.groups,
#     #     #                     width_per_group=self.width_per_group))
#     #     self.in_channel = channel * block.expansion
#     #
#     #     for _ in range(1, block_num):
#     #         layers.append(block(self.in_channel,
#     #                             channel,
#     #                             groups=self.groups,
#     #                             width_per_group=self.width_per_group))
#     #
#     #     return nn.Sequential(*layers)
#
#     def forward(self, x, semantic_prompt):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         B, C, H, W = x.shape
#         semantic_prompt = semantic_prompt.view(B, C, 1, 1)
#         x_cat = torch.cat([x, semantic_prompt], dim=-1)
#         x = self.bottleneck(x_cat)
#         x = self.layer4(x)
#         # x = self.layer5(x, semantic_prompt)
#         # x = self.layer6(x, semantic_prompt)
#         # x = self.layer7(x, semantic_prompt)
#
#         if self.include_top:
#             x = self.avgpool(x)
#             # x = torch.flatten(x, 1)
#             # x = self.fc(x)
#             x = x.view(x.size(0), -1)
#
#         return x
    
class Embedding(nn.Module):
    
    def __init__(self, feature_dim, embed_dim=256, type="ori"):
    
        super(Embedding, self).__init__()
        self.bn = nn.BatchNorm1d(embed_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, embed_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        # print (x.shape)
        x = self.bottleneck(x)
        # print (x.shape)
        if self.type == "bn":
            x = self.bn(x)
        return x
    
class Classifier(nn.Module):
    def __init__(self, embed_dim, class_num, type="linear"):
        super(Classifier, self).__init__()
        
        self.type = type
        if type == 'wn':
            self.fc = nn.utils.weight_norm(nn.Linear(embed_dim, class_num), name="weight")
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(embed_dim, class_num)
            self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x
    

class SFUniDA(nn.Module):
    def __init__(self, args):
        
        super(SFUniDA, self).__init__()
        # self.backbone_layer = ResNet(Bottleneck1, [3, 4, 6, 3], num_classes=1000, include_top=True)
        self.backbone_arch = args.backbone_arch   # resnet50
        self.embed_feat_dim = args.embed_feat_dim # 256
        self.class_num = args.class_num           # shared_class_num + source_private_class_num

        if "resnet" in self.backbone_arch:
            self.backbone_layer = ResBase(self.backbone_arch)
        elif "vgg" in self.backbone_arch:
            self.backbone_layer = VGGBase(self.backbone_arch)
        else:
            raise ValueError("Unknown Feature Backbone ARCH of {}".format(self.backbone_arch))
        
        self.backbone_feat_dim = self.backbone_layer.backbone_feat_dim

        # self.Conv_layer = Conv(self.backbone_feat_dim)

        self.feat_embed_layer = Embedding(self.backbone_feat_dim, self.embed_feat_dim, type="bn")
        
        self.class_layer = Classifier(self.embed_feat_dim, class_num=self.class_num, type="wn")

        text_dim = 512
        self.t2i = torch.nn.Linear(text_dim, 1024, bias=False)
        
    def get_embed_feat(self, input_imgs):
        # input_imgs [B, 3, H, W]
        backbone_feat = self.backbone_layer(input_imgs)
        embed_feat = self.feat_embed_layer(backbone_feat)
        return embed_feat
    
    def forward(self, input_imgs, apply_softmax=True):
        # input_imgs [B, 3, H, W]
        backbone_feat = self.backbone_layer.forward(input_imgs)

        backbone_feat = backbone_feat.view(backbone_feat.size(0), -1)
        
        embed_feat = self.feat_embed_layer(backbone_feat)
        
        cls_out = self.class_layer(embed_feat)
        
        if apply_softmax:
            cls_out = torch.softmax(cls_out, dim=1)
        
        return embed_feat, cls_out

    def forward_with_semantic_prompt(self, input_imgs, semantic_prompt, args, apply_softmax=True):
        # input_imgs [B, 3, H, W]

        # print (semantic_prompt.shape)
        backbone_feat = self.backbone_layer.forward_with_semantic_prompt(input_imgs, semantic_prompt)
        # print (self.backbone_feat_dim)
        # B, C, H, W = backbone_feat.shape
        #
        # semantic_prompt = semantic_prompt.view(B, C, 1, 1)
        # # semantic_prompt = self.Conv_layer(semantic_prompt)
        # backbone_feat = torch.cat([backbone_feat, semantic_prompt], dim=-1)
        # # backbone_feat = backbone_feat.view(backbone_feat.size(0), -1)
        # # print (backbone_feat.shape)
        # backbone_feat = self.Conv_layer(backbone_feat)
        # # backbone_feat = torch.cat([backbone_feat1, backbone_feat], dim=-1)
        # # backbone_feat = self.Conv_layer(backbone_feat)
        # backbone_feat = backbone_feat.view(backbone_feat.size(0), -1)  #[4,2048]
        # print (backbone_feat.shape)

        embed_feat = self.feat_embed_layer(backbone_feat)

        cls_out = self.class_layer(embed_feat)

        if apply_softmax:
            cls_out = torch.softmax(cls_out, dim=1)

        return embed_feat, cls_out