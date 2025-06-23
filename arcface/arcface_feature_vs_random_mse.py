import numpy as np
import torch
import cv2
from torchvision import transforms
from PIL import Image
import os

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

def preprocess(img):
    # 输入img为PIL.Image或np.ndarray
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    tf = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    return tf(img).unsqueeze(0)

def extract_arcface_feature(img):
    x = preprocess(img)
    x = x.cuda() if torch.cuda.is_available() else x
    with torch.no_grad():
        feat = model(x)
        feat = torch.nn.functional.normalize(feat, dim=1)
    return feat.cpu().numpy().flatten()

def main():
    # 读取一张真实图片
    real_img_path = '/home/yu/celeba/img_align_celeba_112/010889.jpg'  # 请替换为实际图片路径
    real_img_path2 = '/home/yu/celeba/img_align_celeba_112/000101.jpg'  # 请替换为实际图片路径
    if not os.path.exists(real_img_path):
        print(f"图片不存在: {real_img_path}")
        return
    real_img = cv2.imread(real_img_path)
    real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)
    real_img2 = cv2.imread(real_img_path2)
    real_img2 = cv2.cvtColor(real_img2, cv2.COLOR_BGR2RGB)
    real_feat = extract_arcface_feature(real_img)
    real_feat2 = extract_arcface_feature(real_img2)

    # 生成一张随机乱码图片
    rand_img = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
    rand_feat = extract_arcface_feature(rand_img)

    # 计算MSE
    mse = np.mean((real_feat - rand_feat) ** 2)
    print(f"ArcFace特征MSE: {mse:.6f}")
    mse = np.mean((real_feat - real_feat2) ** 2)
    print(f"ArcFace特征MSE: {mse:.6f}")

if __name__ == '__main__':
    main()
