import streamlit as st
import json

import pandas as pd

# 读取 stats.jsonl
stats_path = "/home/yu/workspace/mia/training-runs/00037-celeba_feature-cond-ddpmpp-edm-gpus1-batch16-fp32-model1_run/stats.jsonl"
records = []
with open(stats_path, "r") as f:
    for line in f:
        if line.strip().startswith("//"):
            continue
        records.append(json.loads(line))

# 转为 DataFrame
def flatten(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                out[f"{k}/{kk}"] = vv
        else:
            out[k] = v
    return out

df = pd.DataFrame([flatten(r) for r in records])

st.title("训练日志可视化 (stats.jsonl)")

st.write("数据表：")
st.dataframe(df)

# 阿曼（折线图）可视化
metrics = [c for c in df.columns if "/" in c]
selected = st.multiselect("选择要可视化的指标", metrics, default=["Loss/loss/mean", "Resources/peak_gpu_mem_gb/mean"])

for m in selected:
    st.line_chart(df[m], height=200, use_container_width=True)