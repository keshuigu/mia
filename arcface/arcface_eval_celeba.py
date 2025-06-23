import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import random
import cv2
import skimage.transform as sk_transform

# CelebA 路径配置
celeba_root = '/home/yu/celeba/img_align_celeba'  # 图片根目录
identity_path = '/home/yu/celeba/identity_CelebA.txt'  # identity 映射

# 1. 加载 identity 映射
img2pid = {}
with open(identity_path, 'r') as f:
    for line in f:
        img, pid = line.strip().split()
        img2pid[img] = int(pid)

# 2. 构造同/不同人对
img_list = sorted(os.listdir(celeba_root))
# 只保留jpg图片
img_list = [x for x in img_list if x.lower().endswith('.jpg')]
pid2imgs = {}
for img in img_list:
    pid = img2pid.get(img, None)
    if pid is not None:
        pid2imgs.setdefault(pid, []).append(img)

# 构造正样本（同一人两张，所有两两不同组合）
positive_pairs = []
for pid, imgs in pid2imgs.items():
    if len(imgs) >= 2:
        for i in range(len(imgs)):
            for j in range(i+1, len(imgs)):
                if imgs[i] != imgs[j]:
                    positive_pairs.append((imgs[i], imgs[j], 1))

# 构造负样本（不同人各取一张，随机采样，数量与正样本一致）
negative_pairs = []
pid_list = list(pid2imgs.keys())
while len(negative_pairs) < len(positive_pairs):
    pid1, pid2 = random.sample(pid_list, 2)
    img1 = random.choice(pid2imgs[pid1])
    img2 = random.choice(pid2imgs[pid2])
    if img1 != img2:
        negative_pairs.append((img1, img2, 0))

# 合并并打乱
pairs = positive_pairs + negative_pairs
random.shuffle(pairs)

# 3. 加载ArcFace模型
import sys
sys.path.append('/home/yu/workspace/mia/arcface')
from backbones import get_model
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

# 4. 定义transform
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ========== 人脸对齐相关 ========== #
landmark_file = '/home/yu/celeba/list_landmarks_align_celeba.txt'
landmark_dict = {}
if os.path.exists(landmark_file):
    with open(landmark_file, 'r') as f:
        for line in f:
            arr = line.strip().split()
            if len(arr) == 11:
                name = arr[0]
                pts = np.array(list(map(float, arr[1:])), dtype=np.float32).reshape(5, 2)
                landmark_dict[name] = pts

def face_align(img, landmark, image_size=(112, 112)):
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

# 5. 计算相似度并评估（GPU并行化特征提取与相似度计算）
from tqdm import tqdm
batch_size = 64
scores = []
labels = []
pair_names = []

# 先收集所有图片路径，避免重复提取特征
img_set = set()
for img1, img2, _ in pairs[:1000]:
    img_set.add(img1)
    img_set.add(img2)
img_list_unique = sorted(list(img_set))
img2idx = {img: i for i, img in enumerate(img_list_unique)}

# 批量提取所有图片特征（GPU批量+landmark对齐）
features = np.zeros((len(img_list_unique), 512), dtype=np.float32)
batch_size = 64
for i in tqdm(range(0, len(img_list_unique), batch_size), desc='Extract features'):
    batch_imgs = []
    batch_names = img_list_unique[i:i+batch_size]
    for img_name in batch_names:
        img_path = os.path.join(celeba_root, img_name)
        img = Image.open(img_path).convert('RGB')
        # 对齐
        if img_name in landmark_dict:
            aligned = face_align(img, landmark_dict[img_name])
        else:
            print(f'No landmark for {img_name}, using original image.')
            aligned = img
        img_tensor = transform(aligned)
        batch_imgs.append(img_tensor)
    batch_imgs = torch.stack(batch_imgs, dim=0)
    batch_imgs = batch_imgs.cuda() if torch.cuda.is_available() else batch_imgs
    with torch.no_grad():
        feats = model(batch_imgs).cpu().numpy()
        feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
        features[i:i+len(batch_imgs)] = feats

# 计算pair相似度
for img1, img2, label in tqdm(pairs[:1000]):
    if img1 not in img2idx or img2 not in img2idx:
        continue
    idx1 = img2idx[img1]
    idx2 = img2idx[img2]
    feat1 = features[idx1]
    feat2 = features[idx2]
    sim = np.dot(feat1, feat2)
    scores.append(sim)
    labels.append(label)
    # 记录类别信息
    pid1 = img2pid.get(img1, -1)
    pid2 = img2pid.get(img2, -1)
    pair_names.append((img1, img2, label, pid1, pid2))

# 6. 简单阈值评估
best_acc = 0
best_th = 0
all_results = []
for th in np.arange(-1, 1, 0.01):
    preds = [1 if s > th else 0 for s in scores]
    acc = np.mean(np.array(preds) == np.array(labels))
    all_results.append((th, acc))
    if acc > best_acc:
        best_acc = acc
        best_th = th
print(f'CelebA pairs 验证准确率: {best_acc:.4f} (最佳阈值: {best_th:.2f})')

# 7. 输出详细结果到文件
out_path = 'celeba_pair_eval_results.csv'
with open(out_path, 'w') as f:
    f.write('img1,img2,label,pid1,pid2,score,pred,best_th\n')
    for (img1, img2, label, pid1, pid2), score in zip(pair_names, scores):
        pred = 1 if score > best_th else 0
        f.write(f'{img1},{img2},{label},{pid1},{pid2},{score:.6f},{pred},{best_th:.2f}\n')
print(f'详细对比结果已保存到: {out_path}')

# 8. 输出阈值-准确率曲线到文件
curve_path = 'celeba_pair_eval_curve.csv'
with open(curve_path, 'w') as f:
    f.write('threshold,accuracy\n')
    for th, acc in all_results:
        f.write(f'{th:.2f},{acc:.6f}\n')
print(f'阈值-准确率曲线已保存到: {curve_path}')
