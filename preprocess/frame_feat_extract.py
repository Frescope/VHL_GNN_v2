# 使用预训练的googlenet提取每一帧的特征

import torch
from PIL import Image
from torchvision import models, transforms
import numpy as np
import os

# # for TVSum
# FRAME_BASE = r'/data/linkang/tvsum50/frame_2fps/'
# FEATURE_BASE = r'/data/linkang/tvsum50/feature_googlenet_2fps/'

# for SumMe
FRAME_BASE = r'/data/linkang/SumMe/frame_2fps/'
FEATURE_BASE = r'/data/linkang/VHL_GNN/summe_feature_googlenet_2fps/'

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def main():
    if not os.path.isdir(FEATURE_BASE):
        os.makedirs(FEATURE_BASE)

    # functions
    def frame_cmp(name):
        return int(name.split('.jpg')[0])
    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # load images
    data = {}
    for root, dirs, files in os.walk(FRAME_BASE):
        for dir in dirs:
            vid = dir
            frame_names = os.listdir(os.path.join(root,vid))
            frame_names.sort(key = frame_cmp)
            input_tensor = []
            for name in frame_names:
                img = Image.open(os.path.join(root,vid,name))
                img_tensor = preprocess(img)
                img_tensor = img_tensor.unsqueeze(0)
                input_tensor.append(img_tensor)
            input_tensor = torch.cat(input_tensor,0)
            data[vid] = input_tensor
            print(vid,input_tensor.size())

    # extract features
    model = torch.hub.load('pytorch/vision:v0.6.0', 'googlenet', pretrained=True)
    model.eval()
    fmap_block = []
    input_block = []
    def forward_hook(module, data_input, data_output):
        fmap_block.append(data_output)
        input_block.append(data_input)
    model.fc.register_forward_hook(forward_hook)
    model.to('cuda')
    vids = list(data.keys())
    for i in range(len(vids)):
        vid = vids[i]
        print('-'*20,i,vid,'-'*20)
        input_batch = data[vid].to('cuda')
        with torch.no_grad():
            output = model(input_batch)
        features = input_block[0][0].squeeze()
        features = np.array(features.cpu())
        input_block.clear()

        # store features
        feature_path = FEATURE_BASE + vid + '_googlenet_2fps.npy'
        np.save(feature_path,features)
        print(data[vid].size(),features.shape)

    print('GoogleNet Features Extracted !')

if __name__ == '__main__':
    main()