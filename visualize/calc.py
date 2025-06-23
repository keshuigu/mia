import pandas as pd

# 读取csv
df = pd.read_csv('/home/yu/workspace/mia/eval_results_celeba_subset_1000_00038_2103/face_recognition_max_sim.csv')

# 筛选条件
mask = (df['gt_pid'] == df['pred_pid']) & (df['max_sim'] > 0.5)
count = mask.sum()
total = len(df)
ratio = count / total if total > 0 else 0

print(f"gtpid==predpid且maxsim>0.5的数量: {count}")
print(f"总数量: {total}")
print(f"占比: {ratio:.4f}")