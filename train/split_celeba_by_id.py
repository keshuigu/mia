import os
import random
from collections import defaultdict

# 配置
IMG_DIR = '/home/yu/celeba/img_align_celeba_112'
ID_FILE = '/home/yu/celeba/identity_CelebA.txt'
TRAIN_DIR = '/home/yu/celeba/img_align_celeba_112_train'
VAL_DIR = '/home/yu/celeba/img_align_celeba_112_val'
VAL_RATIO = 0.05
SEED = 42

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

# 读取identity映射
img2id = {}
with open(ID_FILE, 'r') as f:
    for line in f:
        img, pid = line.strip().split()
        img2id[img] = int(pid)

# 按id分组
id2imgs = defaultdict(list)
for img, pid in img2id.items():
    id2imgs[pid].append(img)

# 固定随机种子
random.seed(SEED)

# 将所有id随机划分为训练id和验证id，保证id不重叠
all_ids = sorted(id2imgs.keys())
num_val = max(1, int(len(all_ids) * VAL_RATIO))
val_ids = set(random.sample(all_ids, num_val))
train_ids = set(all_ids) - val_ids

train_list = []
val_list = []
for pid in train_ids:
    train_list.extend(id2imgs[pid])
for pid in val_ids:
    val_list.extend(id2imgs[pid])

print(f"总图片数: {len(img2id)}, 训练集: {len(train_list)}, 验证集: {len(val_list)}，训练id: {len(train_ids)}，验证id: {len(val_ids)}")

# 拷贝图片到新目录
from shutil import copyfile
for img in train_list:
    src = os.path.join(IMG_DIR, img)
    dst = os.path.join(TRAIN_DIR, img)
    if not os.path.exists(dst):
        copyfile(src, dst)
for img in val_list:
    src = os.path.join(IMG_DIR, img)
    dst = os.path.join(VAL_DIR, img)
    if not os.path.exists(dst):
        copyfile(src, dst)
print(f"已完成CelebA按id划分训练/验证集，训练集: {TRAIN_DIR}，验证集: {VAL_DIR}")
