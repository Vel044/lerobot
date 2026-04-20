# lerobot/analysis — 实验脚本与图表

`timing_stats.csv` 由 `record.py` 自动追加，所有实验共用（用 `model` 列区分）。
各实验的绘图脚本和输出图存放在对应子目录。

## 目录结构

```
analysis/
├── timing_stats.csv            # record.py 自动写入，全量数据
├── 01_chunk_size_sweep/        # 实验01：chunk_size 参数扫描 ✅
│   ├── plot_timing.py
│   ├── chart1_time_pct.png
│   ├── chart2_fps.png
│   └── chart3_ms_per_frame.png
├── 02_task_comparison/         # 实验02：三类任务工作负载对比 ⬜
├── 03_onnx_optimization/       # 实验03：ONNX 量化推理对比 ⬜
└── README.md
```

## 实验状态

| 编号 | 实验 | 状态 | model 前缀 |
|------|------|------|-----------|
| 01 | chunk_size 参数扫描 | ✅ 已完成 | `so101_act_bottle_cs*` |
| 02 | 三类任务工作负载对比 | ⬜ 待采集 | `so101_act_*_cs100` |
| 03 | ONNX 量化推理对比 | ⬜ 待开始 | `so101_act_onnx_*` |

## 新增实验步骤

1. 在 `timing_stats.csv` 中确认 `model` 命名唯一
2. 在 `analysis/0N_xxx/` 下新建绘图脚本
3. 在 `robotdoc/实验数据/0N_xxx/` 下补充实验设计文档和结论
