import os
import torch
from torchvision import transforms, datasets
from PIL import Image
import numpy as np
import cv2
import skimage.transform as sk_transform

# ========== 人脸对齐相关 ========== #
# LFW官方landmarks路径（如有）
lfw_root = '/home/yu/scikit_learn_data/lfw_home/lfw_funneled'
pairs_path = '/home/yu/scikit_learn_data/lfw_home/pairsDevTest.txt'
landmark_file = '/home/yu/scikit_learn_data/lfw_home/lfw_landmarks_insightface.txt'
landmark_dict = {}
with open(landmark_file, 'r') as f:
    for line in f:
        arr = line.strip().split()
        if len(arr) == 11:
            name = arr[0]
            pts = np.array(list(map(float, arr[1:])), dtype=np.float32).reshape(5, 2)
            landmark_dict[name] = pts

def face_align(img, landmark, image_size=(112, 112)):
    # 标准模板5点
    src = np.array(
        [[38.2946, 51.6963],
         [73.5318, 51.5014],
         [56.0252, 71.7366],
         [41.5493, 92.3655],
         [70.7299, 92.2041]], dtype=np.float32)
    tform = sk_transform.SimilarityTransform()
    tform.estimate(landmark, src)
    M = tform.params[0:2, :]
    aligned = cv2.warpAffine(np.array(img), M, image_size, borderValue=0.0)
    return Image.fromarray(aligned)

# 1. 加载 LFW 数据集和 pairs.txt
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 修正为3通道
])

# 2. 加载模型（使用本地源码和权重）
import sys
sys.path.append('/home/yu/workspace/mia/arcface')
from backbones import get_model

import torch
state_dict = torch.load('/home/yu/workspace/mia/ckpts/arcface_r100.pth', map_location='cpu')
model = get_model('r100')
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
    # 1. 获取landmark
    img_stem = os.path.splitext(os.path.basename(img_path))[0]  # e.g. 'Abel_Pacheco_0001'
    name = img_stem  # 直接用'Abel_Pacheco_0001'作为key
    if name in landmark_dict:
        landmark = landmark_dict[name]
        aligned = face_align(img, landmark)
    else:
        print("fallback to original image, no landmark found for:", name)
        aligned = img  # 检测失败直接用原图
    img = transform(aligned).unsqueeze(0)
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
curve = []
for th in np.arange(-1, 1, 0.01):
    preds = [1 if s > th else 0 for s in scores]
    acc = np.mean(np.array(preds) == np.array(labels))
    curve.append((th, acc))
    if acc > best_acc:
        best_acc = acc
        best_th = th
print(f'LFW pairs 验证准确率: {best_acc:.4f} (最佳阈值: {best_th:.2f})')

# 6. 详细输出到文件
import pandas as pd
results_file = 'lfw_pair_eval_results_r100.csv'
curve_file = 'lfw_pair_eval_curve_r100.csv'
# 每对的详细分数和标签
pair_results = pd.DataFrame({
    'img1': [p[0] for p in pairs],
    'img2': [p[1] for p in pairs],
    'label': labels,
    'score': scores
})
pair_results.to_csv(results_file, index=False)
# 阈值-准确率曲线
curve_df = pd.DataFrame(curve, columns=['threshold', 'accuracy'])
curve_df.to_csv(curve_file, index=False)
print(f'详细结果已保存到: {results_file}, {curve_file}')
