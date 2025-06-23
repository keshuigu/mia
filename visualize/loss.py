import re
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt


def moving_average(data, window_size=10):
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


import numpy as np
from scipy.signal import savgol_filter


def ema(data, alpha=0.01):
    ema_data = []
    for i, x in enumerate(data):
        if i == 0:
            ema_data.append(x)
        else:
            ema_data.append(alpha * x + (1 - alpha) * ema_data[-1])
    return np.array(ema_data)


def parse_loss_file(filename):
    edm_losses = []
    id_losses = []
    total_losses = []#[EDMLossWithIDLossNoGradArcface] edm_loss: 0.0741, id_loss: 0.0008, id_weight: 52.1300, total_loss: 0.1173
    # [EDMLossWithIDLoss] edm_loss: 0.0791, id_loss: 0.3099, total_loss: 7.8263
    # pattern = re.compile(r'\[EDMLossWithIDLoss\].*edm_loss=([\d.]+).*id_loss=([\d.]+).*total_loss=([\d.]+)')
    pattern = re.compile(
        r"\[EDMLossWithIDLossNoGradArcface\].*edm_loss:\s([\d.]+).*id_loss:\s([\d.]+).*total_loss:\s([\d.]+)"
    )

    with open(filename, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                # print(f"Found match: {match.group(0)}")  # Debugging line to see the matched content
                edm_losses.append(float(match.group(1)))
                id_losses.append(float(match.group(2)) * 1000)
                total_losses.append(float(match.group(3)))
    # import itertools
    # edm_losses = list(itertools.accumulate(edm_losses))
    # id_losses = list(itertools.accumulate(id_losses))
    # total_losses = list(itertools.accumulate(total_losses))
    # for i in range(len(edm_losses)):
    #     edm_losses[i] = edm_losses[i] / (i + 1)
    #     id_losses[i] = id_losses[i] / (i + 1)
    #     total_losses[i] = total_losses[i] / (i + 1)
    return edm_losses, id_losses, total_losses


def plot_and_save(edm, idl, total, output="loss_plot.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(edm, label="EDM Loss")
    plt.plot(idl, label="ID Loss")
    # plt.plot(total, label='Total Loss')
    plt.xlabel("Iteration")
    plt.ylabel("Loss Value")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig(output)
    print(f"Plot saved to {output}")


def combat():  # 39 + 44
    edm1, idl1, _ = parse_loss_file(
        "/home/yu/workspace/mia/training-runs/00039-celeba_feature-cond-ddpmpp-edm-gpus1-batch16-fp32-model3_run/log.txt"
    )
    edm2, idl2, _ = parse_loss_file(
        "/home/yu/workspace/mia/training-runs/00044-celeba_feature-cond-ddpmpp-edm-gpus4-batch16-fp32-model3_run/log.txt"
    )
    edm3, idl3, _ = parse_loss_file(
        "/home/yu/workspace/mia/training-runs/00046-celeba_feature-cond-ddpmpp-edm-gpus4-batch16-fp32-model3_run/log.txt"
    )
    import itertools  # 7712.pkl

    edm = edm1[:482000] + edm2 + edm3
    idl = idl1[:482000] + idl2 + idl3
    # edm = savgol_filter(edm, window_length=1001, polyorder=3)
    # idl = savgol_filter(idl, window_length=1001, polyorder=3)
    edm = ema(edm, alpha=0.001)
    idl = ema(idl, alpha=0.001)
    # edm_losses = list(itertools.accumulate(edm))
    # id_losses = list(itertools.accumulate(idl))
    # for i in range(len(edm_losses)):
    #     edm_losses[i] = edm_losses[i] / (i + 1)
    #     id_losses[i] = id_losses[i] / (i + 1)
    plot_and_save(edm, idl, None)


def single():
    # 使用示例
    input_file = "/home/yu/workspace/mia/training-runs/00046-celeba_feature-cond-ddpmpp-edm-gpus4-batch64-fp32/log.txt"  # 替换为你的日志文件路径
    output_image = "46_loss_curve.png"

    edm, idl, total = parse_loss_file(input_file)
    edm = savgol_filter(edm, window_length=1001, polyorder=3)
    idl = savgol_filter(idl, window_length=1001, polyorder=3)
    total = savgol_filter(total, window_length=1001, polyorder=3)
    plot_and_save(edm, idl, total, output_image)


if __name__ == "__main__":
    single()
