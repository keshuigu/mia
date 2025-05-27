import os
import torch
from torchvision import transforms, datasets
from PIL import Image
import numpy as np

# 使用 scikit_learn 下载的数据集路径
lfw_root = '/home/yu/scikit_learn_data/lfw_home/lfw_funneled'
pairs_path = '/home/yu/scikit_learn_data/lfw_home/pairsDevTest.txt'

# 1. 加载 LFW 数据集和 pairs.txt
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# 2. 加载模型（使用本地源码和权重）
import sys
sys.path.append('/home/yu/workspace/mia/arcface')
from arcface.backbones import get_model

import torch
state_dict = torch.load('/home/yu/workspace/mia/ckpts/arcface_r50.pth', map_location='cpu')
model = get_model('r50')
if any(k.startswith('module.') for k in state_dict.keys()):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    state_dict = new_state_dict
model.load_state_dict(state_dict)
model = model.cuda() if torch.cuda.is_available() else model
model.eval()

def get_feature(img_path):
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0)
    img = img.cuda() if torch.cuda.is_available() else img
    with torch.no_grad():
        feat = model(img).cpu().numpy().flatten()
    return feat / np.linalg.norm(feat)

# 3. 解析 pairs.txt
pairs = []
with open(pairs_path, 'r') as f:
    lines = f.readlines()[1:]
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 3:
            name, idx1, idx2 = parts
            img1 = os.path.join(lfw_root, name, f"{name}_{int(idx1):04d}.jpg")
            img2 = os.path.join(lfw_root, name, f"{name}_{int(idx2):04d}.jpg")
            label = 1
        else:
            name1, idx1, name2, idx2 = parts
            img1 = os.path.join(lfw_root, name1, f"{name1}_{int(idx1):04d}.jpg")
            img2 = os.path.join(lfw_root, name2, f"{name2}_{int(idx2):04d}.jpg")
            label = 0
        pairs.append((img1, img2, label))

# 4. 计算相似度并评估
from tqdm import tqdm
scores = []
labels = []
for img1, img2, label in tqdm(pairs):  # 只评估前100对，可去掉[:100]评估全部
    if not os.path.exists(img1) or not os.path.exists(img2):
        continue
    feat1 = get_feature(img1)
    feat2 = get_feature(img2)
    sim = np.dot(feat1, feat2)
    scores.append(sim)
    labels.append(label)

# 5. 简单阈值评估
best_acc = 0
best_th = 0
for th in np.arange(-1, 1, 0.01):
    preds = [1 if s > th else 0 for s in scores]
    acc = np.mean(np.array(preds) == np.array(labels))
    if acc > best_acc:
        best_acc = acc
        best_th = th
print(f'LFW pairs 验证准确率: {best_acc:.4f} (最佳阈值: {best_th:.2f})')
