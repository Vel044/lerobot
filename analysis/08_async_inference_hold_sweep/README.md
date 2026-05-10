# 实验 08 — 异步推理 hold 阈值参数扫描

对应文档：`robotdoc/实验/08_异步推理阈值参数扫描/README.md`

## 文件说明

| 文件 | 说明 |
|------|------|
| `async_client_stats.csv` | `robot_client` 每个 hold episode 的客户端统计 |
| `async_server_stats.csv` | `policy_server` 每次 `predict_action_chunk()` 的推理耗时 |
| `merge_async_stats.py` | 合并 client/server CSV，输出 `async_merged_stats.csv` |
| `plot_async_sweep.py` | 生成汇总表和 4 张图 |
| `async_hold_summary.csv` | 绘图脚本输出的最终汇总表 |
| `chart1_fps_vs_sync.png` | 异步 hold 与同步 baseline 的有效 FPS 对比 |
| `chart2_must_go.png` | hold 与 `must_go` 触发次数 |
| `chart3_frame_breakdown.png` | 执行帧 / 过期帧 / stall 帧组成 |
| `chart4_tinfer_boxplot.png` | `T_infer` 分布箱线图 |

## 复现命令

```bash
cd /Users/vel/Work/RobotOS/Lerobot/lerobot
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate lerobot

python analysis/08_async_inference_hold_sweep/merge_async_stats.py
python analysis/08_async_inference_hold_sweep/plot_async_sweep.py
```

## 采集端输出路径

- `robot_client.py` 默认写入 `~/lerobot/analysis/08_async_inference_hold_sweep/async_client_stats.csv`，
  可用 `LEROBOT_ASYNC_CLIENT_CSV` 覆盖。
- `policy_server.py` 默认写入 `~/lerobot/analysis/08_async_inference_hold_sweep/async_server_stats.csv`，
  可用 `LEROBOT_ASYNC_SERVER_CSV` 覆盖。

## 数据对齐说明

当前这批数据中 `policy_server` 是持续运行的，`async_server_stats.csv` 里的
`episode_idx` 全部为 24，不能直接和 `robot_client` 的 1..7 episode 对齐。
但每组开始时 `timestep` 会回到 0，因此脚本可以用这个 reset 位置恢复 episode 边界。

因此 `merge_async_stats.py` 默认使用 `--match-mode auto`，检测到 episode 不可直接对齐时会
优先按 `timestep` reset 切分；若 reset 数量也不匹配，才退回按采集顺序等分。
后续如果 server 端 episode 编号已修正，可显式使用：

```bash
python merge_async_stats.py --match-mode episode
```

## 注意事项

- `success` 列目前为 `-1`，表示这批 CSV 未写入机器可读的成功/失败标签。
- 动作质量的人工判定记录在 `robotdoc/实验/08_异步推理阈值参数扫描/README.md`。
- 图表使用本目录内数据独立生成，不再依赖顶层 `analysis/plots/`。
