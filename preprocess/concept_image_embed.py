# 将每个concept用图片对应的特征作为嵌入

import torch
from PIL import Image
from torchvision import models, transforms
import numpy as np
import os

CONCEPT_PATH = r'/data/linkang/VHL_GNN/utc/origin_data/Dense_per_shot_tags/Dictionary.txt'
IMAGE_BASE = r'/data/linkang/VHL_GNN/utc/images/'
FEATURE_DIR = r'/data/linkang/VHL_GNN/utc/concept_embeddding/'

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

def main():
    if not os.path.isdir(FEATURE_DIR):
        os.makedirs(FEATURE_DIR)

    # functions
    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def forward_hook(module, data_input, data_output):
        fmap_block.append(data_output)
        input_block.append(data_input)

    # load images
    data = {}
    for root, dirs, files in os.walk(IMAGE_BASE):
        for dir in dirs:
            concept = dir
            if not concept == 'Men':
                continue
            img_names = os.listdir(os.path.join(root,concept))
            input_tensor = []
            img_count = 0
            for name in img_names:
                if not name.split('.')[-1] in ['jpeg', 'webp']:
                    print('Skip: ',name)
                    continue
                try:
                    img = Image.open(os.path.join(root,dir,name))
                    img_tensor = preprocess(img)
                except:
                    print('Error: ',name)
                    continue
                img_tensor = img_tensor.unsqueeze(0)
                input_tensor.append(img_tensor)
                img_count += 1
                if img_count >= 50:
                    break
            input_tensor = torch.cat(input_tensor, 0)
            data[concept] = input_tensor
            print(concept, input_tensor.size())

    # extract features
    model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
    model.eval()
    fmap_block = []
    input_block = []

    model.fc.register_forward_hook(forward_hook)  # 在fc层放置hook
    model.to('cuda')
    concepts = list(data.keys())
    for i in range(len(concepts)):
        concept = concepts[i]
        print(concept, data[concept].size(), end='')
        input_batch = data[concept].to('cuda')
        with torch.no_grad():
            output = model(input_batch)
        features = input_block[0][0].squeeze()
        features = np.array(features.cpu())
        input_block.clear()

        # store features
        feature_path = FEATURE_DIR + concept + '_resnet50.npy'
        np.save(feature_path, features)
        print(features.shape)

if __name__ == '__main__':
    main()