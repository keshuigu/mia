import os
import time
import torch
import numpy as np
from torchvision.utils import make_grid, save_image
import sys

sys.path.append("/home/yu/workspace/mia/edm")
sys.path.append("/home/yu/workspace/mia/arcface")
import dnnlib
import pickle


def load_model(network_pkl, device):
    print(f"Loading model from {network_pkl}...")
    with dnnlib.util.open_url(network_pkl) as f:
        net = pickle.load(f)["ema"].to(device)
    net.eval()
    return net


TARGET = "00040-celeba_feature-cond-ddpmpp-edm-gpus1-batch16-fp32-model2_run"
WATCH_DIR = "/home/yu/workspace/mia/training-runs/" + TARGET
LFW_FEATURE_PATH = "/home/yu/celeba/celeba_arcface_features_align.npz"
OUT_DIR = "/home/yu/workspace/mia/outputs/" + TARGET
os.makedirs(OUT_DIR, exist_ok=True)

# 只取前N个特征用于可视化
NUM_TO_GENERATE = 4
CHECK_INTERVAL = 30  # 秒


def generate_and_save_grid(network_pkl, out_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_model(network_pkl, device)
    lfw_data = np.load(LFW_FEATURE_PATH, allow_pickle=True)
    label_vectors = lfw_data["features"]
    img_names = lfw_data["img_names"] if "img_names" in lfw_data else None
    num_to_generate = min(NUM_TO_GENERATE, label_vectors.shape[0])
    # 生成图片
    images = []
    with torch.no_grad():
        batch_size = num_to_generate
        x_next = torch.randn(
            [batch_size, net.img_channels, net.img_resolution, net.img_resolution],
            device=device,
        )
        labels = torch.tensor(
            label_vectors[:num_to_generate], dtype=torch.float32, device=device
        )
        # 采样参数
        num_steps = 18
        sigma_min = 0.002
        sigma_max = 80
        rho = 7
        S_churn = 0
        S_min = 0
        S_max = float("inf")
        S_noise = 1
        sigma_min_ = max(sigma_min, net.sigma_min)
        sigma_max_ = min(sigma_max, net.sigma_max)
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
        t_steps = (
            sigma_max_ ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min_ ** (1 / rho) - sigma_max_ ** (1 / rho))
        ) ** rho
        t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])
        x_next = x_next.to(torch.float64) * t_steps[0]
        for i, (t_cur, t_next_) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            gamma = (
                min(S_churn / num_steps, np.sqrt(2) - 1)
                if S_min <= t_cur <= S_max
                else 0
            )
            t_hat = net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (
                t_hat**2 - t_cur**2
            ).sqrt() * S_noise * torch.randn_like(x_cur)
            denoised = net(x_hat, t_hat, labels).to(torch.float64)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next_ - t_hat) * d_cur
            if i < num_steps - 1:
                denoised = net(x_next, t_next_, labels).to(torch.float64)
                d_prime = (x_next - denoised) / t_next_
                x_next = x_hat + (t_next_ - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
            del x_cur, x_hat, denoised, d_cur
        images = (x_next * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        images = images.float() / 255.0
    # 拼接保存
    grid = make_grid(images, nrow=num_to_generate, padding=2)
    save_image(grid, out_path)
    print(f"[watch] Saved grid to {out_path}")


def generate_and_save_grid_uncond(network_pkl, out_path, num_to_generate=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_model(network_pkl, device)
    with torch.no_grad():
        batch_size = num_to_generate
        x_next = torch.randn(
            [batch_size, net.img_channels, net.img_resolution, net.img_resolution],
            device=device,
        )
        # 采样参数
        num_steps = 18
        sigma_min = 0.002
        sigma_max = 80
        rho = 7
        S_churn = 0
        S_min = 0
        S_max = float("inf")
        S_noise = 1
        sigma_min_ = max(sigma_min, net.sigma_min)
        sigma_max_ = min(sigma_max, net.sigma_max)
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
        t_steps = (
            sigma_max_ ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min_ ** (1 / rho) - sigma_max_ ** (1 / rho))
        ) ** rho
        t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])
        x_next = x_next.to(torch.float64) * t_steps[0]
        for i, (t_cur, t_next_) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            gamma = (
                min(S_churn / num_steps, np.sqrt(2) - 1)
                if S_min <= t_cur <= S_max
                else 0
            )
            t_hat = net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (
                t_hat**2 - t_cur**2
            ).sqrt() * S_noise * torch.randn_like(x_cur)
            denoised = net(x_hat, t_hat, None).to(torch.float64)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next_ - t_hat) * d_cur
            if i < num_steps - 1:
                denoised = net(x_next, t_next_, None).to(torch.float64)
                d_prime = (x_next - denoised) / t_next_
                x_next = x_hat + (t_next_ - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
            del x_cur, x_hat, denoised, d_cur
        images = (x_next * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        images = images.float() / 255.0
    grid = make_grid(images, nrow=num_to_generate, padding=2)
    save_image(grid, out_path)
    print(f"[watch] Saved unconditional grid to {out_path}")


def watch_and_generate():
    print(f"[watch] Monitoring {WATCH_DIR} for new network-snapshot-*.pkl...")
    seen = set()
    while True:
        files = [
            f
            for f in os.listdir(WATCH_DIR)
            if f.startswith("network-snapshot-003") and f.endswith(".pkl")
        ]
        files.sort()
        for f in files:
            if f not in seen:
                pkl_path = os.path.join(WATCH_DIR, f)
                out_path = os.path.join(OUT_DIR, f.replace(".pkl", "_grid.png"))
                try:
                    generate_and_save_grid(pkl_path, out_path)
                    # 新增：无条件生成
                    # uncond_out_path = out_path.replace("_grid.png", "_uncond_grid.png")
                    # generate_and_save_grid_uncond(pkl_path, uncond_out_path)
                except Exception as e:
                    print(f"[watch] Error processing {f}: {e}")
                seen.add(f)
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    watch_and_generate()
