import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DIR = Path(__file__).parent
df = pd.read_csv(DIR / "timing_stats.csv")

# 从 model 名称提取 chunk_size
df["chunk_size"] = df["model"].str.extract(r"cs(\d+)").astype(int)

# 按 chunk_size 聚合（取均值）
agg = df.groupby("chunk_size")[["obs_pct", "inference_pct", "action_pct", "wait_pct", "frames"]].mean().reset_index()
agg = agg.sort_values("chunk_size")

fig, axes = plt.subplots(2, 1, figsize=(10, 9))

# --- 上图：时间百分比折线图 ---
ax1 = axes[0]
colors = {"obs_pct": "#4C72B0", "inference_pct": "#DD8452", "action_pct": "#55A868", "wait_pct": "#C44E52"}
labels = {"obs_pct": "obs", "inference_pct": "inference", "action_pct": "action", "wait_pct": "wait"}

for col, color in colors.items():
    ax1.plot(agg["chunk_size"], agg[col], marker="o", linewidth=2, color=color, label=labels[col])
    for _, row in agg.iterrows():
        ax1.annotate(f"{row[col]:.1f}%", (row["chunk_size"], row[col]),
                     textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7.5, color=color)

ax1.set_xlabel("Chunk Size", fontsize=11)
ax1.set_ylabel("Time %", fontsize=11)
ax1.set_title("Time breakdown by phase vs Chunk Size (mean over episodes)", fontsize=13)
ax1.set_xticks(agg["chunk_size"])
ax1.set_xticklabels([str(x) for x in agg["chunk_size"]])
ax1.legend(loc="center right", fontsize=9)
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.set_ylim(0, 110)

# --- 下图：每 episode 平均帧数 ---
ax2 = axes[1]
ax2.plot(agg["chunk_size"], agg["frames"], marker="s", linewidth=2, color="#8172B2")
for _, row in agg.iterrows():
    ax2.annotate(f"{row['frames']:.0f}", (row["chunk_size"], row["frames"]),
                 textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)

ax2.set_xlabel("Chunk Size", fontsize=11)
ax2.set_ylabel("Avg frames / episode", fontsize=11)
ax2.set_title("Avg frames per episode vs Chunk Size", fontsize=13)
ax2.set_xticks(agg["chunk_size"])
ax2.set_xticklabels([str(x) for x in agg["chunk_size"]])
ax2.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout(pad=2.5)
out = DIR / "timing_chart.png"
plt.savefig(out, dpi=150)
print(f"saved to {out}")
