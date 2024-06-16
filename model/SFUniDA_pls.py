import torch 
import numpy as np 
import torch.nn as nn
from torchvision import models
from model.mix_prompt import mix_data_prompt
# import clip

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
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.backbone_feat_dim = model_resnet.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

# class Clip(nn.Module):
#     def __init__(self):
#         super(Clip, self).__init__()
#         model_clip, preprocess = clip.load("RN50", device=device)
#         self.encode_image = model_clip.encode_image
#         # self.classifier = nn.Sequential()
#         self.backbone_feat_dim = model_clip.classifier[6].in_features
#
#     def forward(self, x):
#         x = self.encode_image(x)
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

class SaveFeatures():
    features = None

    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output): self.features = output

    def remove(self): self.hook.remove()
    

class SFUniDA_PLS(nn.Module):
    def __init__(self, args, patch_size=(224, 224)):
        
        super(SFUniDA_PLS, self).__init__()
        self.backbone_arch = args.backbone_arch   # resnet50
        self.embed_feat_dim = args.embed_feat_dim # 256
        self.class_num = args.class_num           # shared_class_num + source_private_class_num

        data_prompt = torch.zeros((3, *patch_size))
        self.data_prompt = nn.Parameter(data_prompt)
        # print(self.data_prompt)

        if "resnet" in self.backbone_arch:   
            self.backbone_layer = ResBase(self.backbone_arch) 
        elif "vgg" in self.backbone_arch:
            self.backbone_layer = VGGBase(self.backbone_arch)
        elif "clip" in self.backbone_arch:
            self.backbone_layer = Clip
        else:
            raise ValueError("Unknown Feature Backbone ARCH of {}".format(self.backbone_arch))
        
        self.backbone_feat_dim = self.backbone_layer.backbone_feat_dim
        
        self.feat_embed_layer = Embedding(self.backbone_feat_dim, self.embed_feat_dim, type="bn")
        
        self.class_layer = Classifier(self.embed_feat_dim, class_num=self.class_num, type="wn")

        self.bn_f = [SaveFeatures(self.feat_embed_layer.bn), SaveFeatures(self.backbone_layer.bn1),
                     SaveFeatures(self.backbone_layer.layer1[0].conv1), SaveFeatures(self.backbone_layer.layer1[0].conv2), SaveFeatures(self.backbone_layer.layer1[0].conv3),
                     SaveFeatures(self.backbone_layer.layer1[1].conv1), SaveFeatures(self.backbone_layer.layer1[1].conv2), SaveFeatures(self.backbone_layer.layer1[1].conv3),
                     SaveFeatures(self.backbone_layer.layer1[2].conv1), SaveFeatures(self.backbone_layer.layer1[2].conv2), SaveFeatures(self.backbone_layer.layer1[2].conv3),
                     SaveFeatures(self.backbone_layer.layer2[0].conv1), SaveFeatures(self.backbone_layer.layer2[0].conv2), SaveFeatures(self.backbone_layer.layer2[0].conv3),
                     SaveFeatures(self.backbone_layer.layer2[1].conv1), SaveFeatures(self.backbone_layer.layer2[1].conv2), SaveFeatures(self.backbone_layer.layer2[1].conv3),
                     SaveFeatures(self.backbone_layer.layer2[2].conv1), SaveFeatures(self.backbone_layer.layer2[2].conv2), SaveFeatures(self.backbone_layer.layer2[2].conv3),
                     SaveFeatures(self.backbone_layer.layer2[3].conv1), SaveFeatures(self.backbone_layer.layer2[3].conv2), SaveFeatures(self.backbone_layer.layer2[3].conv3),
                     SaveFeatures(self.backbone_layer.layer3[0].conv1), SaveFeatures(self.backbone_layer.layer3[0].conv2), SaveFeatures(self.backbone_layer.layer3[0].conv3),
                     SaveFeatures(self.backbone_layer.layer3[1].conv1), SaveFeatures(self.backbone_layer.layer3[1].conv2), SaveFeatures(self.backbone_layer.layer3[1].conv3),
                     SaveFeatures(self.backbone_layer.layer3[2].conv1), SaveFeatures(self.backbone_layer.layer3[2].conv2), SaveFeatures(self.backbone_layer.layer3[2].conv3),
                     SaveFeatures(self.backbone_layer.layer3[3].conv1), SaveFeatures(self.backbone_layer.layer3[3].conv2), SaveFeatures(self.backbone_layer.layer3[3].conv3),
                     SaveFeatures(self.backbone_layer.layer3[4].conv1), SaveFeatures(self.backbone_layer.layer3[4].conv2), SaveFeatures(self.backbone_layer.layer3[4].conv3),
                     SaveFeatures(self.backbone_layer.layer3[5].conv1), SaveFeatures(self.backbone_layer.layer3[5].conv2), SaveFeatures(self.backbone_layer.layer3[5].conv3),
                     SaveFeatures(self.backbone_layer.layer4[0].conv1), SaveFeatures(self.backbone_layer.layer4[0].conv2), SaveFeatures(self.backbone_layer.layer4[0].conv3),
                     SaveFeatures(self.backbone_layer.layer4[1].conv1), SaveFeatures(self.backbone_layer.layer4[1].conv2), SaveFeatures(self.backbone_layer.layer4[1].conv3),
                     SaveFeatures(self.backbone_layer.layer4[2].conv1), SaveFeatures(self.backbone_layer.layer4[2].conv2), SaveFeatures(self.backbone_layer.layer4[2].conv3),]

        # for name, param in self.backbone_layer.named_parameters():
        #     param.requires_grad = False
        # for name, param in self.feat_embed_layer.named_parameters():
        #     param.requires_grad = False


        
    def get_embed_feat(self, input_imgs):
        # input_imgs [B, 3, H, W]
        backbone_feat = self.backbone_layer(input_imgs)
        embed_feat = self.feat_embed_layer(backbone_feat)
        return embed_feat
    
    def forward(self, input_imgs, apply_softmax=True):
        # input_imgs [B, 3, H, W]
        backbone_feat = self.backbone_layer(input_imgs)
        # print (backbone_feat.shape)
        
        embed_feat = self.feat_embed_layer(backbone_feat)
        
        cls_out = self.class_layer(embed_feat)
        
        if apply_softmax:
            cls_out = torch.softmax(cls_out, dim=1)
        
        return embed_feat, cls_out

    def forward_pls(self, input_imgs, apply_softmax=True):
        # print(self.data_prompt)
        input_imgs = mix_data_prompt(input_imgs, self.data_prompt)

        backbone_feat = self.backbone_layer(input_imgs)
        embed_feat = self.feat_embed_layer(backbone_feat)
        cls_out = self.class_layer(embed_feat)

        if apply_softmax:
            cls_out = torch.softmax(cls_out, dim=1)

        return embed_feat, cls_out, self.bn_f