import os
from tqdm import tqdm 
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

def train_transform(resize_size=256, crop_size=224,):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]) 
    
def test_transform(resize_size=256, crop_size=224,):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])
    
'''
assume classes across domains are the same.
[0 1 ............................................................................ N - 1]
|---- common classes --||---- source private classes --||---- target private classes --|

|-------------------------------------------------|
|                DATASET PARTITION                |
|-------------------------------------------------|
|DATASET    |  class split(com/sou_pri/tar_pri)   |
|-------------------------------------------------|
|DATASET    |    PDA    |    OSDA    |   UniDA    |
|-------------------------------------------------|
|Office-31  |  10/21/0  |  10/0/11   |  10/10/11  |
|-------------------------------------------------|
|OfficeHome |  25/40/0  |  25/0/40   |  10/5/50   |
|-------------------------------------------------|
|VisDA-C    |           |   6/0/6    |   6/3/3    |
|-------------------------------------------------|  
|DomainNet  |           |            | 150/50/145 |
|-------------------------------------------------|
'''

class SFUniDADataset(Dataset):
    
    def __init__(self, args, data_dir, data_list, d_type, preload_flg=True) -> None:
        super(SFUniDADataset, self).__init__()
        
        self.d_type = d_type
        self.dataset = args.dataset
        self.preload_flg = preload_flg
        self.idx2text = {}
        
        self.shared_class_num = args.shared_class_num
        self.source_private_class_num = args.source_private_class_num
        self.target_private_class_num = args.target_private_class_num 
        
        self.shared_classes = [i for i in range(args.shared_class_num)]
        self.source_private_classes = [i + args.shared_class_num for i in range(args.source_private_class_num)]
        
        if args.dataset == "Office" and args.target_label_type == "OSDA":
            self.target_private_classes = [i + args.shared_class_num + args.source_private_class_num + 10 for i in range(args.target_private_class_num)]
        else:
            self.target_private_classes = [i + args.shared_class_num + args.source_private_class_num for i in range(args.target_private_class_num)]
            
        self.source_classes = self.shared_classes + self.source_private_classes
        self.target_classes = self.shared_classes + self.target_private_classes
        
        self.data_dir = data_dir 
        self.data_list = [item.strip().split() for item in data_list]
        # print (len(data_list))
        
        # Filtering the data_list
        if self.d_type == "source":
            # self.data_dir = args.source_data_dir
            self.data_list = [item for item in self.data_list if int(item[1]) in self.source_classes]
            for item in self.data_list:
                if int(item[1]) in self.source_classes:
                    # self.data_list = item
                    idx = int(item[1])
                    text = item[0].split('/')
                    text = text[0]
                    self.idx2text[idx] = text
            # for item in self.data_list:
            #     length = [int(item[1])]
            #     # length = length.replace(' ', ',')
            #     # print (length)
            #     text = item[0].split('/')
            #     text = text[0]
            #     # for i in length:
            #     #     self.idx2text[i] = text[i]
            #     # text[length] = text
            #     # i = 0
            #     # if text != False :
            #     #     text1[length(i++)] = text(i++)
            #     # print (text.count('back_pack'))
            #     # text = [x for x in text if text.count(x) == 1]
            #     # print (length)
            #     # for idx in length:
            #     #     text, _ = self.data_list[idx]
            #     #     text, _ = text.split('/')
            #     # print (text)
            #     for idx in length:
            #         self.idx2text[idx] = text
            #     print (self.idx2text)
            # if args.dataset == "Office":
            #     if args.s_idx == 0:
            #         with open('./data/Office/Amazon/Amazon_OPDA_idx2text.txt', 'r') as f:
            #             for line in f.readlines():
            #                 idx, text= line.strip().split()
            #                 idx = idx.strip(':')
            #                 text = text.replace('_', ' ')
            #                 # text, _ = text.split('/')
            #                 # print (text)
            #                 self.idx2text[idx] = text
            #                 print (self.idx2text[2])
            #     elif args.s_idx == 1:
            #         with open('./data/Office/Dslr/Dslr_idx2text.txt', 'r') as f:
            #             for line in f.readlines():
            #                 idx, text = line.strip().split()
            #                 idx = idx.strip(':')
            #                 text = text.replace('_', ' ')
            #                 self.idx2text[idx] = text
            #     elif args.s_idx == 2:
            #         with open('./data/Office/Webcam/Webcam_idx2text.txt', 'r') as f:
            #             for line in f.readlines():
            #                 idx, text = line.strip().split()
            #                 idx = idx.strip(':')
            #                 text = text.replace('_', ' ')
            #                 self.idx2text[idx] = text
            # elif args.dataset == "OfficeHome":
            #     with open('./data/OfficeHome/image_unida_list.txt', 'r') as f:
            #         for line in f.readlines():
            #             idx, text = line.strip().split()
            #             idx = idx.strip(':')
            #             text = text.replace('_', ' ')
            #             self.idx2text[idx] = text
            # elif args.dataset == "VisDA":
            #     with open('./data/VisDA/train/image_unida_list.txt', 'r') as f:
            #         for line in f.readlines():
            #             idx, text = line.strip().split()
            #             idx = idx.strip(':')
            #             text = text.replace('_', ' ')
            #             self.idx2text[idx] = text
        else:
            # self.data_dir = args.target_data_dir
            self.data_list = [item for item in self.data_list if int(item[1]) in self.target_classes]
            
        self.pre_loading()
        
        self.train_transform = train_transform()
        self.test_transform = test_transform()



    def pre_loading(self):
        if "Office" in self.dataset and self.preload_flg:
            self.resize_trans = transforms.Resize((256, 256))
            print("Dataset Pre-Loading Started ....")
            self.img_list = [self.resize_trans(Image.open(os.path.join(self.data_dir, item[0])).convert("RGB")) for item in tqdm(self.data_list, ncols=60)]
            print("Dataset Pre-Loading Done!")
        else:
            pass
    
    def load_img(self, img_idx):
        img_f, img_label = self.data_list[img_idx]
        if "Office" in self.dataset and self.preload_flg:
            img = self.img_list[img_idx]
        else:
            img = Image.open(os.path.join(self.data_dir, img_f)).convert("RGB")        
        return img, img_label
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, img_idx):
        
        img, img_label = self.load_img(img_idx)
        # text = img_label
        
        if self.d_type == "source":
            img_label = int(img_label)
        else:
            img_label = int(img_label) if int(img_label) in self.source_classes else len(self.source_classes)
        
        img_train = self.train_transform(img)
        img_test = self.test_transform(img)

        # text = self.dataset.source_classes[img_label]
        text = self.idx2text[img_label]
        text = 'A photo of a ' + text
        # print (text)

        return img_train, img_test, img_label, img_idx, text
    