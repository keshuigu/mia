# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import PIL.Image
import torch
from torch_utils import persistence

#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# 条件特征嵌入版EDM Loss（适用于ArcFace等特征作为条件标签）
@persistence.persistent_class
class EDMFeatureCondLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, perceptual_weight=0, identity_weight=0.5, arcface_ckpt_path='/home/yu/workspace/mia/ckpts/arcface_r50.pth'):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.perceptual_weight = perceptual_weight
        self.identity_weight = identity_weight
        # 加载VGG16感知损失网络
        import torchvision.models as models
        vgg = models.vgg16(pretrained=True).features.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.vgg_layers = [3, 8, 15, 22]  # relu1_2, relu2_2, relu3_3, relu4_3
        # 加载ArcFace模型（假设512维，r50/r100均可）
        import sys
        sys.path.append('/home/yu/workspace/mia/arcface')
        from backbones import get_model
        self.arcface = get_model('r50', fp16=False).eval()
        if arcface_ckpt_path is not None:
            ckpt = torch.load(arcface_ckpt_path, map_location='cpu')
            self.arcface.load_state_dict(ckpt)
        for p in self.arcface.parameters():
            p.requires_grad = False

    def perceptual_loss(self, x, y):
        # x, y: [B, C, H, W], 归一化到[0,1]，3通道
        def get_features(img):
            self.vgg = self.vgg.to(img.device)  # 保证VGG和输入在同一device
            feats = []
            h = img
            for i, layer in enumerate(self.vgg):
                h = layer(h)
                if i in self.vgg_layers:
                    feats.append(h)
            return feats
        feats_x = get_features(x)
        feats_y = get_features(y)
        loss = 0
        for fx, fy in zip(feats_x, feats_y):
            loss = loss + torch.nn.functional.mse_loss(fx, fy)
        return loss

    def identity_loss(self, gen_img, target_feat):
        # gen_img: [B, C, H, W]，target_feat: [B, 512]
        # 归一化到[0,1]，转为3通道
        if gen_img.shape[1] == 1:
            img = gen_img.repeat(1,3,1,1)
        else:
            img = gen_img
        img = (img + 1) / 2  # [-1,1] -> [0,1]
        # ArcFace预处理（BGR, 112x112, 标准化）
        img = torch.nn.functional.interpolate(img, size=(112,112), mode='bilinear', align_corners=False)
        img = img[:, [2,1,0], :, :]  # RGB->BGR
        mean = torch.tensor([0.5,0.5,0.5], device=img.device).view(1,3,1,1)
        std = torch.tensor([0.5,0.5,0.5], device=img.device).view(1,3,1,1)
        img = (img - mean) / std
        self.arcface = self.arcface.to(img.device)  # 保证ArcFace和输入在同一device
        with torch.no_grad():
            feat = self.arcface(img).detach()
        feat = torch.nn.functional.normalize(feat, dim=1)
        target_feat = torch.nn.functional.normalize(target_feat, dim=1)
        # 余弦距离loss
        loss = 1 - (feat * target_feat).sum(dim=1).mean()
        return loss

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        # EDM loss
        loss = weight * ((D_yn - y) ** 2)
        # 感知损失（图像需归一化到[0,1]且为3通道）
        if y.shape[1] == 1:
            y_vgg = y.repeat(1,3,1,1)
            D_yn_vgg = D_yn.repeat(1,3,1,1)
        else:
            y_vgg = y
            D_yn_vgg = D_yn
        y_vgg = (y_vgg + 1) / 2  # [-1,1] -> [0,1]
        D_yn_vgg = (D_yn_vgg + 1) / 2
        perceptual = self.perceptual_loss(D_yn_vgg, y_vgg)
        # identity loss
        identity = 0
        if labels is not None:
            identity = self.identity_loss(D_yn, labels)
        total_loss = loss.mean() + self.perceptual_weight * perceptual + self.identity_weight * identity
        return total_loss

#----------------------------------------------------------------------------
# 条件特征嵌入版EDM Loss（支持人脸mask加权）
@persistence.persistent_class
class EDMFeatureCondMaskLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, perceptual_weight=0.2, identity_weight=0.4, mask_weight=1.0, bg_weight=0.0, arcface_ckpt_path='/home/yu/workspace/mia/ckpts/arcface_r50.pth'):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.perceptual_weight = perceptual_weight
        self.identity_weight = identity_weight
        self.mask_weight = mask_weight  # 人脸区域loss权重
        self.bg_weight = bg_weight      # 背景区域loss权重
        # VGG16感知损失网络
        import torchvision.models as models
        vgg = models.vgg16(pretrained=True).features.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.vgg_layers = [3, 8, 15, 22]
        # ArcFace模型
        import sys
        sys.path.append('/home/yu/workspace/mia/arcface')
        from backbones import get_model
        self.arcface = get_model('r50', fp16=False).eval()
        if arcface_ckpt_path is not None:
            ckpt = torch.load(arcface_ckpt_path, map_location='cpu')
            self.arcface.load_state_dict(ckpt)
        for p in self.arcface.parameters():
            p.requires_grad = False

    def perceptual_loss(self, x, y, mask=None):
        # x, y: [B, C, H, W], mask: [B,1,H,W] or None
        def get_features(img):
            self.vgg = self.vgg.to(img.device)
            feats = []
            h = img
            for i, layer in enumerate(self.vgg):
                h = layer(h)
                if i in self.vgg_layers:
                    feats.append(h)
            return feats
        feats_x = get_features(x)
        feats_y = get_features(y)
        loss = 0
        for fx, fy in zip(feats_x, feats_y):
            if mask is not None:
                m = torch.nn.functional.interpolate(mask, size=fx.shape[2:], mode='bilinear', align_corners=False)
                loss = loss + ((fx - fy) ** 2 * m).sum() / (m.sum() + 1e-8)
            else:
                loss = loss + torch.nn.functional.mse_loss(fx, fy)
        return loss

    def identity_loss(self, gen_img, target_feat, mask=None):
        # gen_img: [B, C, H, W]，target_feat: [B, 512]，mask: [B,1,H,W] or None
        if gen_img.shape[1] == 1:
            img = gen_img.repeat(1,3,1,1)
        else:
            img = gen_img
        img = (img + 1) / 2
        if mask is not None:
            mask_resized = torch.nn.functional.interpolate(mask, size=(img.shape[2], img.shape[3]), mode='bilinear', align_corners=False)
            img = img * mask_resized + (1 - mask_resized) * 0.5  # 背景填充0.5
        img = torch.nn.functional.interpolate(img, size=(112,112), mode='bilinear', align_corners=False)
        img = img[:, [2,1,0], :, :]
        mean = torch.tensor([0.5,0.5,0.5], device=img.device).view(1,3,1,1)
        std = torch.tensor([0.5,0.5,0.5], device=img.device).view(1,3,1,1)
        img = (img - mean) / std
        self.arcface = self.arcface.to(img.device)  # 保证ArcFace和输入在同一device
        with torch.no_grad():
            feat = self.arcface(img).detach()
        feat = torch.nn.functional.normalize(feat, dim=1)
        target_feat = torch.nn.functional.normalize(target_feat, dim=1)
        # 余弦距离loss
        loss = 1 - (feat * target_feat).sum(dim=1).mean()
        return loss

    def get_face_mask(self, images):
        """
        使用mediapipe SelfieSegmentation自动生成人脸mask，无需本地权重。
        输入: images [B, C, H, W]，范围[-1,1]或[0,1]，3通道
        输出: mask [B,1,H,W]，人脸区域为1，背景为0
        """
        import mediapipe as mp
        import numpy as np
        import torch
        masks = []
        mp_selfie_segmentation = mp.solutions.selfie_segmentation
        seg = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
        for img in images:
            img_np = img.detach().cpu().numpy()
            if img_np.min() < 0:
                img_np = (img_np + 1) / 2
            img_np = np.transpose(img_np, (1, 2, 0))  # [H,W,C]
            img_np = (img_np * 255).astype(np.uint8)
            res = seg.process(img_np)
            mask = res.segmentation_mask  # [H,W], float32, 0~1
            mask = (mask > 0.5).astype(np.float32)
            masks.append(torch.from_numpy(mask).unsqueeze(0))  # [1,H,W]
        masks = torch.stack(masks, dim=0)  # [B,1,H,W]
        return masks.to(images.device)

    def __call__(self, net, images, labels=None, mask=None, augment_pipe=None):
        # mask: [B,1,H,W]，值为1表示人脸，0为背景
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        # mask加权像素loss
        if mask is not None:
            mask = mask.to(y.device)
            pixel_weight = mask * self.mask_weight + (1 - mask) * self.bg_weight
            loss = (weight * ((D_yn - y) ** 2) * pixel_weight).sum() / (pixel_weight.sum() + 1e-8)
        else:
            loss = (weight * ((D_yn - y) ** 2)).mean()
        # 感知损失
        if y.shape[1] == 1:
            y_vgg = y.repeat(1,3,1,1)
            D_yn_vgg = D_yn.repeat(1,3,1,1)
        else:
            y_vgg = y
            D_yn_vgg = D_yn
        y_vgg = (y_vgg + 1) / 2
        D_yn_vgg = (D_yn_vgg + 1) / 2
        perceptual = self.perceptual_loss(D_yn_vgg, y_vgg, mask=mask)
        # identity loss
        identity = 0
        if labels is not None:
            identity = self.identity_loss(D_yn, labels, mask=mask)
        total_loss = loss + self.perceptual_weight * perceptual + self.identity_weight * identity
        return total_loss

#----------------------------------------------------------------------------
# EDMFeatureCondMixupDropoutLoss，支持label dropout和label mixup
@persistence.persistent_class
class EDMFeatureCondMixupDropoutLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, identity_weight=5, arcface_ckpt_path='/home/yu/workspace/mia/ckpts/arcface_r100.pth', label_dropout_prob=0.2, label_mixup_alpha=0.4):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.identity_weight = identity_weight
        self.label_dropout_prob = label_dropout_prob
        self.label_mixup_alpha = label_mixup_alpha
        # ArcFace模型
        import sys
        sys.path.append('/home/yu/workspace/mia/arcface')
        from backbones import get_model
        import torch
        self.arcface = get_model('r100', fp16=False).eval()
        if arcface_ckpt_path is not None:
            ckpt = torch.load(arcface_ckpt_path, map_location='cpu')
            self.arcface.load_state_dict(ckpt)
        for p in self.arcface.parameters():
            p.requires_grad = False
        # 人脸对齐工具
        self._face_tools_initialized = False
        self.face_app = None

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'face_app' in state:
            state.pop('face_app', None)
        state['_face_tools_initialized'] = False
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.face_app = None
        self._face_tools_initialized = False

    def _lazy_init_face_tools(self):
        if not self._face_tools_initialized:
            from insightface.app import FaceAnalysis
            import torch
            self.face_app = FaceAnalysis(allowed_modules=['detection'])
            self.face_app.prepare(ctx_id=0 if torch.cuda.is_available() else -1)
            self._face_tools_initialized = True

    def align_face(self, img_tensor):
        import numpy as np
        import cv2
        self._lazy_init_face_tools()
        img = img_tensor.detach().cpu().numpy()
        img = (img + 1) / 2  # [0,1]
        img = np.transpose(img, (1,2,0)) * 255
        img = img.astype(np.uint8)
        faces = self.face_app.get(img)
        if len(faces) == 0:
            return None  # 检测失败
        landmark = faces[0].kps.astype(np.float32)  # (5,2)
        src = np.array(
            [[38.2946, 51.6963],
             [73.5318, 51.5014],
             [56.0252, 71.7366],
             [41.5493, 92.3655],
             [70.7299, 92.2041]], dtype=np.float32)
        from skimage.transform import SimilarityTransform
        tform = SimilarityTransform()
        tform.estimate(landmark, src)
        M = tform.params[0:2, :]
        aligned = cv2.warpAffine(img, M, (112, 112), borderValue=0.0)
        aligned = aligned.astype(np.float32) / 255.0
        return aligned

    def identity_loss(self, gen_img, target_feat):
        import torch
        import numpy as np
        import cv2
        aligned_imgs = []
        for img in gen_img:
            aligned = self.align_face(img)
            if aligned is not None:
                aligned_imgs.append(aligned)
            else:
                # 检测失败用中心裁剪
                self._lazy_init_face_tools()
                img_np = img.detach().cpu().numpy()
                img_np = (img_np + 1) / 2
                img_np = np.transpose(img_np, (1,2,0))
                img_np = cv2.resize((img_np*255).astype(np.uint8), (112,112))
                aligned_imgs.append(img_np.astype(np.float32)/255.0)
        aligned_imgs = np.stack(aligned_imgs)  # [B,112,112,3]
        # ArcFace预处理: BGR, [0,1]->[-1,1], NCHW
        imgs_bgr = aligned_imgs[..., ::-1]
        imgs_bgr = (imgs_bgr - 0.5) / 0.5
        imgs_bgr = torch.from_numpy(imgs_bgr).permute(0,3,1,2).float()
        imgs_bgr = imgs_bgr.to(target_feat.device)
        self.arcface = self.arcface.to(imgs_bgr.device)
        with torch.no_grad():
            feats = self.arcface(imgs_bgr).detach()
        feats = torch.nn.functional.normalize(feats, dim=1)
        target_feat = torch.nn.functional.normalize(target_feat, dim=1)
        loss = 1 - (feats * target_feat).sum(dim=1).mean()
        return loss

    def label_dropout(self, labels):
        if labels is None:
            return None
        device = labels.device
        mask = (torch.rand(labels.shape[0], device=device) > self.label_dropout_prob).float().unsqueeze(1)
        return labels * mask

    def label_mixup(self, images, labels):
        import numpy as np
        if labels is None:
            return images, labels
        batch_size = images.shape[0]
        if batch_size < 2:
            return images, labels
        lam = np.random.beta(self.label_mixup_alpha, self.label_mixup_alpha)
        index = torch.randperm(batch_size, device=images.device)
        mixed_images = lam * images + (1 - lam) * images[index]
        mixed_labels = lam * labels + (1 - lam) * labels[index]
        return mixed_images, mixed_labels

    def __call__(self, net, images, labels=None, augment_pipe=None):
        # label dropout
        labels = self.label_dropout(labels)
        # label mixup
        images, labels = self.label_mixup(images, labels)
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = (weight * ((D_yn - y) ** 2)).mean()
        # identity loss
        identity = 0
        if labels is not None:
            identity = self.identity_loss(D_yn, labels)
        total_loss = loss + self.identity_weight * identity
        print(f"[EDMFeatureCondMixupDropoutLoss] edm_loss: {loss.item():.4f}, id_loss: {identity if isinstance(identity, (int,float)) else identity.item():.4f}, total_loss: {total_loss.item():.4f}")
        return total_loss

#----------------------------------------------------------------------------
# EDMLoss变种，增加ID Loss，自动检测人脸landmark并对齐
@persistence.persistent_class
class EDMLossWithIDLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, id_weight=10, arcface_ckpt_path='/home/yu/workspace/mia/ckpts/arcface_r100.pth'):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.id_weight = id_weight
        # ArcFace模型
        import sys
        sys.path.append('/home/yu/workspace/mia/arcface')
        from backbones import get_model
        import torch
        state_dict = torch.load(arcface_ckpt_path, map_location='cpu')
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
        self.arcface = model  # ArcFace模型，512维特征
        # 多卡兼容：ArcFace始终与输入imgs_bgr同device
        self._face_tools_initialized = False  # 避免insightface等被pickle

    def __getstate__(self):
        state = self.__dict__.copy()
        if hasattr(self, 'face_app'):
            state.pop('face_app', None)
        state['_face_tools_initialized'] = False
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.face_app = None
        self._face_tools_initialized = False

    def _lazy_init_face_tools(self):
        if not self._face_tools_initialized:
            from insightface.app import FaceAnalysis
            import torch
            self.face_app = FaceAnalysis(allowed_modules=['detection','landmark'])
            self.face_app.prepare(ctx_id=0 if torch.cuda.is_available() else -1)
            self._face_tools_initialized = True

    def align_face(self, img_tensor):
        import numpy as np
        import cv2
        import PIL.Image
        self._lazy_init_face_tools()
        img = img_tensor.detach().cpu().numpy()
        img = (img + 1) / 2  # [0,1]
        img = np.transpose(img, (1,2,0)) * 255
        img = img.clip(0, 255)  # 确保像素值在[0,255]
        PIL.Image.fromarray(img.astype(np.uint8)).save('face_align_input.jpg')  # debug
        img = img.astype(np.uint8)
        faces = self.face_app.get(img)
        if len(faces) == 0:
            return None  # 检测失败
        landmark = faces[0].kps.astype(np.float32)  # (5,2)
        # 标准模板
        src = np.array(
            [[38.2946, 51.6963],
             [73.5318, 51.5014],
             [56.0252, 71.7366],
             [41.5493, 92.3655],
             [70.7299, 92.2041]], dtype=np.float32)
        from skimage.transform import SimilarityTransform
        tform = SimilarityTransform()
        tform.estimate(landmark, src)
        M = tform.params[0:2, :]
        aligned = cv2.warpAffine(img, M, (112, 112), borderValue=0.0)
        aligned = aligned.astype(np.float32) / 255.0
        return aligned

    def id_loss(self, gen_imgs, data_feats):
        import torch
        import numpy as np
        import cv2
        aligned_imgs = []
        for img in gen_imgs:
            aligned = self.align_face(img)
            if aligned is not None:
                aligned_imgs.append(aligned)
            else:
                # 检测失败用中心裁剪
                self._lazy_init_face_tools()
                img_np = img.detach().cpu().numpy()
                img_np = (img_np + 1) / 2
                img_np = np.transpose(img_np, (1,2,0))
                img_np = cv2.resize((img_np*255).astype(np.uint8), (112,112))
                aligned_imgs.append(img_np.astype(np.float32)/255.0)
        aligned_imgs = np.stack(aligned_imgs)  # [B,112,112,3]
        # ArcFace预处理: BGR, [0,1]->[-1,1], NCHW
        imgs_bgr = aligned_imgs[..., ::-1]
        imgs_bgr = (imgs_bgr - 0.5) / 0.5
        imgs_bgr = torch.from_numpy(imgs_bgr).permute(0,3,1,2).float()
        imgs_bgr = imgs_bgr.to(data_feats.device)  # 保证与labels同device
        self.arcface = self.arcface.to(imgs_bgr.device)
        with torch.no_grad():
            feats = self.arcface(imgs_bgr).detach()
        feats = torch.nn.functional.normalize(feats, dim=1)
        data_feats = torch.nn.functional.normalize(data_feats, dim=1)
        fac = 0.7
        loss = fac * (1 - (feats * data_feats).sum(dim=1).mean()) + (1 - fac) * torch.nn.functional.mse_loss(feats, data_feats).mean()
        return loss


    def id_loss_2(self, gen_imgs, data_feats):
        with torch.no_grad():
            feats = self.arcface(gen_imgs).detach()
        feats = torch.nn.functional.normalize(feats, dim=1)
        data_feats = torch.nn.functional.normalize(data_feats, dim=1)
        fac = 1
        loss = fac * (1 - (feats * data_feats).sum(dim=1).mean()) + (1 - fac) * torch.nn.functional.mse_loss(feats, data_feats).mean()
        return loss

    def __call__(self, net, images, labels=None, augment_pipe=None):
        # labels: [B,512] ArcFace特征
        # debug
        print(self.id_loss_2(images, labels).item())
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        edm_loss = weight * ((D_yn - y) ** 2)
        edm_loss = edm_loss.mean()
        id_loss = 0
        if labels is not None:
            id_loss = self.id_loss(D_yn, labels)
        total_loss = edm_loss + self.id_weight * id_loss
        print(f"[EDMLossWithIDLoss] edm_loss: {edm_loss.item():.4f}, id_loss: {id_loss if isinstance(id_loss, (int,float)) else id_loss.item():.4f}, total_loss: {total_loss.item():.4f}")
        return total_loss



@persistence.persistent_class
class EDMLossWithIDLossWithoutAlign:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, id_weight=10, arcface_ckpt_path='/home/yu/workspace/mia/ckpts/arcface_r100.pth'):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.id_weight = id_weight
        # ArcFace模型
        import sys
        sys.path.append('/home/yu/workspace/mia/arcface')
        from backbones import get_model
        import torch
        state_dict = torch.load(arcface_ckpt_path, map_location='cpu')
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
        self.arcface = model  # ArcFace模型，512维特征
        # 多卡兼容：ArcFace始终与输入imgs_bgr同device


    def id_loss(self, gen_imgs, data_feats):
        with torch.no_grad():
            feats = self.arcface(gen_imgs).detach()
        feats = torch.nn.functional.normalize(feats, dim=1)
        data_feats = torch.nn.functional.normalize(data_feats, dim=1)
        fac = 0.7
        loss = fac * (1 - (feats * data_feats).sum(dim=1).mean()) + (1 - fac) * torch.nn.functional.mse_loss(feats, data_feats).mean()
        return loss

    def __call__(self, net, images, labels=None, augment_pipe=None):
        # labels: [B,512] ArcFace特征
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        edm_loss = weight * ((D_yn - y) ** 2)
        edm_loss = edm_loss.mean()
        id_loss = 0
        if labels is not None:
            id_loss = self.id_loss(D_yn, labels)
        total_loss = edm_loss + self.id_weight * id_loss
        print(f"[EDMLossWithIDLoss] edm_loss: {edm_loss.item():.4f}, id_loss: {id_loss if isinstance(id_loss, (int,float)) else id_loss.item():.4f}, total_loss: {total_loss.item():.4f}")
        return total_loss

#----------------------------------------------------------------------------
# EDMLossWithIDLossNoGradArcface类，ArcFace参数冻结但允许主网络通过ArcFace获得梯度（不加no_grad），用于identity loss。
@persistence.persistent_class
class EDMLossWithIDLossNoGradArcface:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, id_weight=100, arcface_ckpt_path='/home/yu/workspace/mia/ckpts/arcface_r100.pth', id_weight_schedule=None, id_weight_warmup_steps=10000):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.id_weight = id_weight
        self.id_weight_schedule = id_weight_schedule  # 可传入自定义调度函数
        self.id_weight_warmup_steps = id_weight_warmup_steps  # 线性warmup步数
        # ArcFace模型
        import sys
        sys.path.append('/home/yu/workspace/mia/arcface')
        from backbones import get_model
        import torch
        state_dict = torch.load(arcface_ckpt_path, map_location='cpu')
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
        for p in model.parameters():
            p.requires_grad = False  # 明确冻结参数
        self.arcface = model  # ArcFace模型，512维特征

    def get_id_weight(self, cur_step=None, cur_epoch=None):
        # 优先自定义调度
        if self.id_weight_schedule is not None:
            return self.id_weight_schedule(cur_step, cur_epoch)
        # 默认线性warmup: 从0到self.id_weight
        if cur_step is not None and self.id_weight_warmup_steps > 0:
            w = min(float(cur_step) / self.id_weight_warmup_steps, 1.0)
            return self.id_weight * w
        return self.id_weight

    def id_loss(self, gen_imgs, data_feats):
        # 不加with torch.no_grad()，允许主网络获得梯度，但ArcFace参数冻结
        feats = self.arcface(gen_imgs)
        feats = torch.nn.functional.normalize(feats, dim=1)
        data_feats = torch.nn.functional.normalize(data_feats, dim=1)
        loss = torch.nn.functional.mse_loss(feats, data_feats).mean()
        return loss

    def __call__(self, net, images, labels=None, augment_pipe=None, cur_step=None, cur_epoch=None):
        # labels: [B,512] ArcFace特征
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        edm_loss = weight * ((D_yn - y) ** 2)
        edm_loss = edm_loss.mean()
        id_loss = 0
        if labels is not None:
            id_loss = self.id_loss(D_yn, labels)
        cur_id_weight = self.get_id_weight(cur_step=cur_step, cur_epoch=cur_epoch)
        total_loss = edm_loss + cur_id_weight * id_loss
        print(f"[EDMLossWithIDLossNoGradArcface] edm_loss: {edm_loss.item():.4f}, id_loss: {id_loss if isinstance(id_loss, (int,float)) else id_loss.item():.4f}, id_weight: {cur_id_weight:.4f}, total_loss: {total_loss.item():.4f}")
        return total_loss
