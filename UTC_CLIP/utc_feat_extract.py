# 使用预训练的CLIP模型提取帧特征

import clip
import torch
import pickle
import numpy as np
from PIL import Image

FRAME_DIR = r'/data/linkang/VHL_GNN/utc/frames/'
FEATURE_BASE = r'/data/linkang/VHL_GNN/utc/features/'
CONCEPT_PATH = r'/data/linkang/VHL_GNN/utc/origin_data/Dense_per_shot_tags/Dictionary.txt'

SHOTS_NUMS = [2783, 3692, 2152, 3588]
D_OUTPUT = 512

def getVisualFeatures(frame_dir, feature_base):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    with torch.no_grad():
        for vid in range(1, 5):
            visual_features = []
            frame_base = frame_dir + 'P0%d/' % vid
            image_list = []
            for i in range(SHOTS_NUMS[vid - 1] * 5):
                path = frame_base + str(i).zfill(5) + '.jpg'
                image = preprocess(Image.open(path)).unsqueeze(0).to(device)
                image_list.append(image)
                if len(image_list) >= 200:
                    images = torch.cat(image_list, 0)
                    image_list.clear()
                    image_feature = model.encode_image(images)
                    visual_features.append(image_feature.cpu().numpy())

                if i % 1000 == 0 and i > 0:
                    print(i, len(visual_features))

            images = torch.cat(image_list, 0)
            image_list.clear()
            image_feature = model.encode_image(images)
            visual_features.append(image_feature.cpu().numpy())

            vid_features = np.concatenate(visual_features, axis=0)
            visual_features.clear()
            np.save(feature_base + 'V%d_CLIP.npy' % vid, vid_features)
            print('CLIP Features: ', vid, vid_features.shape)

    return

def getConceptFeatures(concept_path):
    concepts = []
    with open(concept_path, 'r') as f:
        for word in f.readlines():
            concepts.append(word.strip().split("'")[1])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    with torch.no_grad():
        concept_embs = {}
        for word in concepts:
            word_emb = clip.tokenize(word).to(device)
            word_feature = model.encode_text(word_emb)
            concept_embs[word] = word_feature.cpu().numpy()

        with open(r'/data/linkang/VHL_GNN/utc/concept_clip.pkl', 'wb') as file:
            pickle.dump(concept_embs, file)

    return


if __name__ == '__main__':
    # getVisualFeatures(FRAME_DIR, FEATURE_BASE)
    # getConceptFeatures(CONCEPT_PATH)

    with open(r'/data/linkang/VHL_GNN/utc/concept_clip.pkl','rb') as file:
        concepts_embs = pickle.load(file)
    print()