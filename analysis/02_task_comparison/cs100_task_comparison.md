# 三任务时间占比对比（cs100）

数据来源：`timing_stats.csv`  
比较对象：bottle_pick (cs100) vs bottle_push (默认=cs100) vs classification (默认=cs100)  
仅看百分比，屏蔽 episode 总时长差异的干扰。

---

## 汇总表

| 阶段 | bottle_pick (n=20) | bottle_push (n=10) | classification (n=4) |
|------|-------------------|-------------------|---------------------|
| **obs** | 15.4% (12.4–18.1) | 13.9% (11.8–16.5) | 12.1% (8.8–13.8) |
| **inference** | 36.5% (33.7–44.5) | 34.5% (31.6–38.5) | **39.6%** (34.3–50.3) |
| **action** | 4.8% (4.1–5.2) | 4.6% (4.2–5.1) | 3.9% (2.7–4.6) |
| **wait** | 43.3% (38.6–48.2) | **47.0%** (41.5–51.2) | 44.4% (38.2–47.7) |

---

## 关键结论

### 1. inference 占比：classification > pick > push
- classification 的推理占比最高（39.6%），比 pick（36.5%）高 3.1pp，比 push（34.5%）高 5.1pp
- classification 的 inference 方差最大（ep0 高达 50.3%），episode 间抖动明显
- **原因推断（修正）**：三者 cs 相同，模型体量一致。classification episode 长达 60s，树莓派5长时间满载后 CPU 热降频，导致后段推理变慢，整体 inference 占比被拉高。这是**硬件热管理问题**，而非模型本身更重。pick（30s）和 push（40s）episode 较短，降频影响较小。
- 验证方向：对比 classification 各 episode 内前半段 vs 后半段的单帧推理时延，若后半段明显变长则确认为降频所致。

### 2. wait 占比：push 最高
- push 的 wait 占比（47.0%）明显高于 pick（43.3%）和 classification（44.4%）
- wait 代表机器人执行动作 chunk 的等待时间，push 动作更慢/行程更长，导致系统经常空等
- 这说明 push 任务的瓶颈在**执行侧**，而非推理侧；适当减小 cs 可能有效

### 3. obs 占比：pick > push > classification
- 三者差距不大（12–15%），classification 略低
- classification episode 更长（60s），obs 绝对时间更多，但占比被更高的 inference 和 wait 稀释

### 4. action（写舵机）占比：三者几乎一致（~4-5%）
- 写动作本身耗时稳定，与任务类型无关

---

## 图示（文字版堆叠图）

```
                obs     inference   action  wait
pick (cs100)  ████░░  ██████████████░  █░  ███████████████░
push (cs100)  ███░░░  █████████████░░  █░  ████████████████░
classif(cs100)███░░░  ███████████████░  █░  ████████████████░
              0%      ~13%          ~50%   ~55%            100%
```

---

## 建议

| 任务 | 瓶颈 | 优化方向 |
|------|------|---------|
| classification | episode 长（60s）→ CPU 热降频拉高 inference 占比 | 加强散热；或拆分为多个短 episode；监控 CPU 频率确认 |
| push | wait 占比最高，执行侧慢 | 尝试减小 cs（如 cs50），减少每次等待长度 |
| pick | 较均衡 | 当前 cs100 配置合理，无明显单一瓶颈 |
