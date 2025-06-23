import json
import matplotlib.pyplot as plt

# 文件名
logfile = '/home/yu/workspace/mia/training-runs/00037-celeba_feature-cond-ddpmpp-edm-gpus1-batch16-fp32-model1_run/stats.jsonl'  # 你的日志文件名

# 逐行读取并解析
records = []
with open(logfile, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            records.append(obj)
        except Exception as e:
            print(f"[WARN] 跳过无法解析的行: {e}")

if not records:
    raise RuntimeError('未解析到任何有效数据')

# 收集所有包含loss的key
all_keys = set()
for rec in records:
    all_keys.update([k for k in rec.keys() if 'loss' in k.lower()])
if not all_keys:
    raise RuntimeError('未找到包含loss的key')

# 按key绘制曲线
plt.figure(figsize=(10,6))
for key in sorted(all_keys):
    y = [rec.get(key, {}).get('mean', None) for rec in records]
    # 过滤无效
    x = list(range(len(y)))
    y = [v if v is not None else float('nan') for v in y]
    plt.plot(x, y, label=key)
plt.xlabel('Step/Index')
plt.ylabel('Loss')
plt.title('Loss Curve (from JSON lines)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('celeba_pair_eval_loss_curve.png')
print('已保存loss曲线为 celeba_pair_eval_loss_curve.png')
