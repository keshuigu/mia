import pandas as pd
import matplotlib.pyplot as plt

# 读取阈值-准确率曲线数据
curve_path = 'celeba_pair_eval_curve.csv'
df = pd.read_csv(curve_path)

plt.figure(figsize=(8, 5))
plt.plot(df['threshold'], df['accuracy'], marker='o')
plt.xlabel('Threshold')
plt.ylabel('Accuracy')
plt.title('CelebA Pairs Threshold-Accuracy Curve')
plt.grid(True)
plt.tight_layout()
plt.savefig('celeba_pair_eval_curve.png')
plt.show()
