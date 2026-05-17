# ECBIT · 科研任务计划
## Hermes Agent 执行文档 · ERA5-Conditioned Block Imputation Transformer

**论文标题**：ERA5-Conditioned Block Imputation for Sparse Antarctic Weather Station Time Series

**目标期刊**：IEEE Transactions on Geoscience and Remote Sensing（TGRS）或 Remote Sensing（MDPI）

**启动时机**：AntAWS benchmarking 论文投稿后立即启动

---

## 总体时间线（4周）

```
Week 1  │ 数据工程：AntAWS 站点筛选 + ERA5 复用对齐 + 缺失模式统计分析
Week 2  │ 模型实现：ECBIT 架构 + 基线模型 + 训练框架
Week 3  │ 实验执行（服务器）‖ 论文框架 + Introduction + Related Work（并行）
Week 4  │ 结果分析 + 图表生成 + 论文主体撰写 + 精修投稿
```

**并行原则**：服务器跑实验期间，Hermes 同步推进论文撰写，不空转等待。

---

## Part 1 · 项目根目录文件

### `AGENTS.md`

```markdown
# AGENTS.md — ECBIT Research Project

## Project Goal
Complete and submit "ERA5-Conditioned Block Imputation for Sparse Antarctic
Weather Station Time Series" to IEEE TGRS or Remote Sensing.
No hard DDL — target submission within 4 weeks of project start.

## Project Context (Critical Background)
This project builds directly on two prior experiments:
1. AntAWS Benchmarking paper (completed): iTransformer best for forecasting,
   variate-token attention captures T/P→wind physical coupling.
2. ECAFT experiment (failed): ERA5 met vars redundant with AWS (r=0.90-1.00),
   ERA5 RH has systematic bias (-10 to -25%), naive cross-attention fails.

ECBIT's design is a DIRECT RESPONSE to ECAFT's failure:
- ERA5 as CONDITION (not parallel stream) → only activates for missing vars
- Block missing training → matches real polar sensor failure patterns
- Variable-level gating → ERA5 RH bias only affects RH imputation

## Data Assets (Already Available — Do NOT Re-download)
- AntAWS raw CSVs: data/antaws/raw/  (267 stations, 1980-2021, 3h)
- ERA5 aligned data: data/era5/  (from ECAFT project, 5 met vars, 2h)
  → ERA5 is at 2h resolution; AntAWS is at 3h → need resampling to 3h
- IMAU AWS NetCDF: data/imau/  (19 stations, 1995-2022, 2h, optional)

## Compute Routing (MANDATORY)
- LOCAL (32-thread CPU, NO GPU, conda darts):
  → data preprocessing, station analysis, script writing,
    LaTeX compilation, matplotlib/seaborn plotting, result analysis.
  → NEVER run train.py or imputation scripts locally.
- REMOTE (GPU cluster, conda darts with DL libs):
  → ALL model training and evaluation.
  → Use microclimate-experiment-server SKILL for ALL job submissions.

## Parallel Work Protocol
After submitting a remote experiment batch:
  → Immediately switch to paper writing or literature tasks.
  → Poll experiment status every 2 hours.
  → Pull results as soon as jobs complete; update findings.md immediately.

## SKILL Usage
- /planning-with-files:plan  → read at EVERY session start
- autoresearch               → literature search (imputation, polar AWS)
- ml-paper-writing           → LaTeX writing, IEEE/TGRS format
- academic-plotting          → matplotlib figures, TikZ diagrams
- weights-and-biases         → experiment tracking (project: ecbit-polar)
- microclimate-experiment-server → remote job submission

## Paper Toolchain
- Writing:  LaTeX (IEEE two-column for TGRS, paper/main.tex)
- Diagrams: TikZ (paper/figures/)
- Plots:    Python matplotlib/seaborn (local darts, scripts/plot_*.py)
- Compile:  latexmk -pdf paper/main.tex

## Error Recovery
If BLOCKING error:
1. Run scripts/release_resources.sh
2. clawbot send "ECBIT BLOCKED: <error>. Action: <action>"
3. Log in progress.md under ## Blocked Issues
4. Stop cleanly

## Git Convention
Format: [phase] action: description
e.g.:  [data] feat: AntAWS station selection, 38 stations retained
       [model] feat: ECBIT conditional cross-attention module
       [exp] result: Round 1 baselines complete, SAITS MAE=X.XX
       [paper] write: Methodology section 3.1-3.4
```

### `CLAUDE.md`

```markdown
# CLAUDE.md — Claude Code Session Configuration

## Mandatory Session Start
1. Read agent/task_plan.md — identify current task.
2. Read last 5 entries in agent/progress.md.
3. Invoke /planning-with-files:plan.
4. ONE atomic task per session. No scope expansion.

## Mandatory Session End
1. Mark completed task [x] in task_plan.md.
2. Append 3-5 bullet summary to progress.md.
3. Append key findings to findings.md if applicable.
4. Run acceptance criteria command.
5. git add -A && git commit -m "[phase] action: description"
6. If experiment batch submitted: clawbot send status.
7. If experiment batch complete: immediately start parallel paper task.

## Key Design Decisions (Do NOT change without explicit instruction)
- Block missing simulation: min_block=6, max_block=72 steps (3h resolution)
  → 6 steps = 18h (short freeze), 72 steps = 9 days (winter maintenance)
- ERA5 conditioning: variable-level gating (missing_vars mask)
  → ERA5 only influences variables that have missing timesteps
  → Observable variables are NOT touched by ERA5
- iTransformer backbone: variate-token (C=5 tokens, each = full time series)
  → Motivated by AntAWS benchmarking paper's attention analysis
- Loss: MSE on missing positions only (not on observed positions)
- Normalization: Z-score per variable, fit on train split

## Forbidden Locally
- python src/train.py  (use microclimate-experiment-server)
- Any training or large-scale inference script

## W&B Convention
Project: ecbit-polar
Run naming: {model}_{station_group}_{miss_pattern}_{miss_rate}_s{seed}
Log: MAE/RMSE per variable (T, RH, wspd, P, q) + mean
```

### `agent/task_plan.md`（初始版）

```markdown
# ECBIT Task Plan
Goal: Submit ERA5-Conditioned Block Imputation paper to IEEE TGRS / Remote Sensing.
Current Phase: Phase 0 — Initialization
Last Updated: [DATE]

---

## Phase 0 · Initialization (Day 1)
- [ ] Create project directory structure (see AGENTS.md for layout)
- [ ] Write AGENTS.md, CLAUDE.md, README.md
- [ ] git init, first commit: "[init] ECBIT project scaffold"
- [ ] Verify all SKILLs installed and functional
- [ ] Test microclimate-experiment-server connectivity (ping remote)
- [ ] Test ClawBot: clawbot send "ECBIT project initialized"
- [ ] Initialize task_plan.md / findings.md / progress.md

## Phase 1 · Data Engineering (Days 2–4)
- [ ] AntAWS station selection script (completeness filter: T>70%, ≥10yr record)
      → Target: 30-40 main stations + 5 held-out stations
      → Output: data/station_meta_ecbit.csv
      Acceptance: CSV exists, station count 30-45, held-out clearly labeled

- [ ] ERA5 resampling: resample existing 2h ERA5 to 3h (match AntAWS)
      → Script: src/resample_era5.py
      → Output: data/era5_3h/<station_id>.npz
      Acceptance: all selected stations have ERA5 3h files, no NaN

- [ ] Missing pattern analysis script
      → Script: scripts/analyze_missing_patterns.py (local darts)
      → Output: results/missing_analysis/pattern_stats.csv
      → Compute: mean block length, max block length, block count per station/var
      Acceptance: CSV exists, findings written to findings.md

- [ ] AntAWS preprocessing pipeline for imputation
      → Script: src/preprocess_antaws_impute.py
      → Output: data/processed/<station_id>_impute.npz
         Fields: X_full (complete segments), E_3h (ERA5), T_enc (time encoding)
      → Split: 70% train / 10% val / 20% test (temporal, no leakage)
      Acceptance: unit test passes (shape, no NaN in complete segments)

- [ ] Block missing simulator unit test
      → Script: src/utils/block_missing.py
      → Test: simulate_block_missing() produces correct block structure
      Acceptance: pytest src/tests/test_block_missing.py passes

## Phase 2 · Model Implementation (Days 5–8)
- [ ] Implement src/models/ecbit.py (full ECBIT model)
      Acceptance: forward pass (B=4, T=168, C=5) → output (B, T, C), no error
                  unit test: ERA5 gating — observable vars unchanged by ERA5

- [ ] Implement src/baselines/linear_interp.py (linear interpolation)
      Acceptance: fills missing positions, preserves observed values

- [ ] Implement src/baselines/locf.py (Last Observation Carried Forward)
      Acceptance: correct LOCF behavior on block missing test case

- [ ] Implement src/baselines/saits_wrapper.py (SAITS via PyPOTS)
      Acceptance: PyPOTS SAITS trains and evaluates on AntAWS format

- [ ] Implement src/baselines/brits_wrapper.py (BRITS via PyPOTS)
      Acceptance: PyPOTS BRITS trains and evaluates on AntAWS format

- [ ] Implement src/baselines/itransformer_impute.py (iTransformer for imputation)
      Acceptance: forward pass correct, loss only on missing positions

- [ ] Implement src/baselines/era5_direct.py (ERA5 direct substitution)
      Acceptance: fills missing with ERA5 values (after bias correction on train set)

- [ ] Implement src/train_impute.py (unified training entry, YAML config)
      Acceptance: smoke test — 1 station, 10 epochs, result JSON written

- [ ] Implement src/evaluate_impute.py (MAE/RMSE per variable + mean)
      Acceptance: correct metrics on synthetic test case

- [ ] Generate all experiment YAML configs
      → experiments/configs/round1/ (6 baselines × 3 patterns × 3 rates × 3 seeds = 162)
      → experiments/configs/round2/ (4 ECBIT variants × 3 patterns × 3 rates × 3 seeds = 108)
      → experiments/configs/round3/ (ECBIT full × 5 held-out × 3 patterns × 3 seeds = 45)
      Acceptance: ls experiments/configs/round1/ | wc -l == 162

## Phase 3 · Experiments (Days 9–13, parallel with paper writing)

### Round 1 — Baseline Evaluation (Days 9–10)
- [ ] Submit Round 1: 162 baseline jobs to remote server
      → clawbot send "ECBIT Round 1 submitted: 162 baseline jobs"
      → [IMMEDIATELY] Switch to paper writing: Introduction + Related Work

- [ ] [PARALLEL] Write paper/sections/introduction.tex
      → AntAWS缺失问题, 块缺失vs点缺失, ERA5条件化动机, 贡献声明
      Acceptance: latexmk compiles, ~0.6 pages

- [ ] [PARALLEL] Write paper/sections/related_work.tex
      → Imputation methods (SAITS, BRITS, CSDI), polar AWS data, ERA5 bias correction
      Acceptance: latexmk compiles, ~0.6 pages

- [ ] Pull Round 1 results, generate results/tables/baseline_table.csv
      → clawbot send "ECBIT Round 1 done. Best baseline: <model> MAE=X.XX"
      Acceptance: 162 JSON files present, baseline_table.csv generated

### Round 2 — ECBIT Ablation (Days 11–12)
- [ ] Submit Round 2: 108 ECBIT ablation jobs
      → clawbot send "ECBIT Round 2 submitted: 108 ablation jobs"
      → [IMMEDIATELY] Switch to paper: Methodology §3.1–3.4

- [ ] [PARALLEL] Write paper/sections/methodology.tex §3.1–3.3
      → Problem formulation, block missing simulation, ECBIT architecture
      Acceptance: latexmk compiles, ~1.0 pages

- [ ] [PARALLEL] Generate Fig.1: ECBIT architecture diagram (TikZ)
      → paper/figures/fig1_ecbit_arch.tex
      Acceptance: compiles in main.tex, readable at double-column width

- [ ] Pull Round 2 results, generate results/tables/ablation_table.csv
      → Key check: ECBIT full < ECBIT no-ERA5 on long-block missing
      → clawbot send "ECBIT Round 2 done. Full vs no-ERA5: X.XX vs X.XX"
      Acceptance: ablation_table.csv generated, findings written to findings.md

### Round 3 — Generalization (Day 13)
- [ ] Submit Round 3: 45 held-out station jobs
      → clawbot send "ECBIT Round 3 submitted: 45 generalization jobs"
      → [IMMEDIATELY] Switch to paper: Methodology §3.4–3.5

- [ ] [PARALLEL] Write paper/sections/methodology.tex §3.4–3.5
      → ERA5 conditional cross-attention, variable-level gating, training objective
      Acceptance: latexmk compiles, methodology section complete

- [ ] Pull Round 3 results, generate results/tables/generalization_table.csv
      Acceptance: 45 JSON files present, generalization_table.csv generated

## Phase 4 · Analysis + Visualization (Days 14–16)
- [ ] Generate Fig.2: Missing pattern analysis figure
      → scripts/plot_missing_patterns.py (local darts)
      → Bar chart: mean block length per variable per station group
      → Comparison: AntAWS real patterns vs MCAR assumption
      Acceptance: paper/figures/fig2_missing_patterns.pdf, 300 DPI

- [ ] Generate Fig.3: Main results comparison figure
      → scripts/plot_main_results.py (local darts)
      → Line plot: MAE vs block length (short/medium/long) for all models
      → Separate panels for high-missingness vs low-missingness stations
      Acceptance: paper/figures/fig3_main_results.pdf, 300 DPI

- [ ] Generate Fig.4: ERA5 conditioning effect analysis
      → scripts/plot_era5_effect.py (local darts)
      → Scatter: ERA5-AWS RH bias (x) vs ECBIT gain over no-ERA5 (y), per station
      → Expected: positive correlation (larger bias → larger gain from conditioning)
      Acceptance: paper/figures/fig4_era5_effect.pdf, 300 DPI

- [ ] Generate Fig.5: Imputation quality visualization
      → scripts/plot_imputation_curves.py (local darts)
      → Time series: ground truth vs ECBIT vs SAITS vs ERA5-direct
      → Select: Byrd station, wind speed variable, long-block missing segment
      → Annotate: missing block boundaries, ERA5 reference line
      Acceptance: paper/figures/fig5_imputation_curves.pdf, 300 DPI

## Phase 5 · Paper Writing (Days 17–22)
- [ ] Write paper/sections/experiments.tex §4.1–4.2
      → Setup table, main results Table I (baselines), analysis text
      Acceptance: latexmk compiles, Table I populated from baseline_table.csv

- [ ] Write paper/sections/experiments.tex §4.3–4.5
      → Ablation Table II, ERA5 conditioning analysis, generalization Table III
      Acceptance: latexmk compiles, all tables populated from CSV files

- [ ] Write paper/sections/conclusion.tex
      → Summary, limitations (ERA5 RH bias, station count), future work
      Acceptance: latexmk compiles, ~0.4 pages

- [ ] Write Abstract (≤250 words for TGRS)
      → Include: dataset, task, method name, key result numbers
      Acceptance: word count ≤250, all key numbers present

- [ ] Full paper review: logic, citations, formatting
      → Use ml-paper-writing SKILL citation verification workflow
      → Check: all figures referenced in text, all tables cited
      Acceptance: latexmk compiles cleanly, no undefined references

- [ ] IEEE TGRS format compliance check
      → Double-column, Times New Roman 10pt, page count ≤10
      → All figures ≥300 DPI, font ≥8pt
      → References IEEE format
      Acceptance: PDF passes format check

- [ ] Final PDF + submission
      → clawbot send "ECBIT paper ready for submission"

---
## Blocked Issues
(none)

## Completion Criteria
Paper submitted to IEEE TGRS or Remote Sensing.
```

---

## Part 2 · 项目目录结构

```
ecbit/
├── data/
│   ├── antaws/
│   │   ├── raw/              # AntAWS 原始 CSV（已有，从 ECAFT 项目复用）
│   │   └── processed/        # 预处理后 .npz（X_full, E_3h, T_enc）
│   ├── era5/                 # ERA5 2h 数据（已有，从 ECAFT 项目复用）
│   ├── era5_3h/              # ERA5 重采样到 3h（新增）
│   ├── imau/                 # IMAU AWS NetCDF（已有，可选）
│   └── station_meta_ecbit.csv
├── src/
│   ├── utils/
│   │   ├── block_missing.py  # 块缺失模拟器
│   │   └── time_encoding.py  # 时间编码（月份、小时、cos_sza、极昼标志）
│   ├── models/
│   │   └── ecbit.py          # 完整 ECBIT 模型
│   ├── baselines/
│   │   ├── linear_interp.py
│   │   ├── locf.py
│   │   ├── saits_wrapper.py  # PyPOTS SAITS
│   │   ├── brits_wrapper.py  # PyPOTS BRITS
│   │   ├── itransformer_impute.py
│   │   └── era5_direct.py
│   ├── dataset.py            # AntAWSImputeDataset
│   ├── preprocess_antaws_impute.py
│   ├── resample_era5.py
│   ├── train_impute.py
│   └── evaluate_impute.py
├── scripts/
│   ├── analyze_missing_patterns.py
│   ├── plot_missing_patterns.py
│   ├── plot_main_results.py
│   ├── plot_era5_effect.py
│   ├── plot_imputation_curves.py
│   └── release_resources.sh
├── experiments/
│   ├── configs/
│   │   ├── round1/           # 162 YAML（6基线）
│   │   ├── round2/           # 108 YAML（4 ECBIT 变体）
│   │   └── round3/           # 45 YAML（泛化测试）
│   └── results/
│       ├── metrics/          # JSON 结果文件
│       └── tables/           # 汇总 CSV
├── paper/
│   ├── main.tex
│   ├── sections/
│   │   ├── introduction.tex
│   │   ├── related_work.tex
│   │   ├── methodology.tex
│   │   ├── experiments.tex
│   │   └── conclusion.tex
│   ├── figures/
│   └── references.bib
├── agent/
│   ├── task_plan.md
│   ├── findings.md
│   └── progress.md
├── AGENTS.md
├── CLAUDE.md
└── README.md
```

---

## Part 3 · 模型核心代码

### `src/utils/block_missing.py`

```python
import torch
import numpy as np

def simulate_block_missing(
    x: torch.Tensor,
    miss_rate: float = 0.3,
    min_block: int = 6,
    max_block: int = 72,
    per_variable: bool = True
) -> torch.Tensor:
    """
    模拟极地 AWS 的结构性块缺失（传感器失效驱动）。

    Args:
        x           : (T, C) 完整序列
        miss_rate   : 目标缺失率（0.1–0.6）
        min_block   : 最小连续缺失步数（默认6步=18h at 3h resolution）
        max_block   : 最大连续缺失步数（默认72步=9天）
        per_variable: True=每个变量独立采样（单传感器失效）
                      False=所有变量同时缺失（数据记录器失效）
    Returns:
        mask : (T, C) 二值掩码，1=缺失
    """
    T, C = x.shape
    mask = torch.zeros(T, C, dtype=torch.float32)

    channels = range(C) if per_variable else [0]
    for c in channels:
        t = 0
        while t < T:
            if torch.rand(1).item() < miss_rate:
                block_len = torch.randint(min_block, max_block + 1, (1,)).item()
                end = min(t + block_len, T)
                if per_variable:
                    mask[t:end, c] = 1.0
                else:
                    mask[t:end, :] = 1.0
                t = end
            else:
                t += 1
    return mask


def get_real_missing_mask(x_raw: torch.Tensor) -> torch.Tensor:
    """从原始 AntAWS 数据提取真实缺失掩码（NaN 位置）。"""
    return torch.isnan(x_raw).float()


def apply_mask(x: torch.Tensor, mask: torch.Tensor,
               fill_value: float = 0.0) -> torch.Tensor:
    """将掩码应用到序列，缺失位置填充 fill_value。"""
    x_masked = x.clone()
    x_masked[mask.bool()] = fill_value
    return x_masked
```

### `src/models/ecbit.py`

```python
import torch
import torch.nn as nn
import math


class VariateTokenEncoder(nn.Module):
    """
    iTransformer 风格的 variate-token 编码器。
    每个变量的完整时间序列作为一个 token。

    Input:  x : (B, T, C_in)  — C_in = n_vars + n_mask + n_time
    Output: z : (B, C_vars, d_model)
    """
    def __init__(self, seq_len: int, n_vars: int, n_aux: int,
                 d_model: int, n_heads: int, n_layers: int,
                 d_ff: int, dropout: float = 0.1):
        super().__init__()
        # 每个 variate token = 该变量的完整时间序列 + 辅助特征
        self.var_proj = nn.Linear(seq_len * (1 + n_aux // n_vars + 1), d_model)
        # 简化：直接将 (T, C_in) reshape 后按变量投影
        self.input_proj = nn.Linear(seq_len, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, n_vars, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) → transpose → (B, C, T) → project → (B, C, d_model)
        B, T, C = x.shape
        z = self.input_proj(x.transpose(1, 2))  # (B, C, d_model)
        z = z + self.pos_emb[:, :C, :]
        return self.encoder(z)                   # (B, C, d_model)


class ConditionalCrossAttention(nn.Module):
    """
    变量级条件化交叉注意力。
    仅对有缺失的变量激活 ERA5 条件信息，可观测变量不受影响。

    Q = Z_obs (AWS 观测流)
    K/V = Z_era5 (ERA5 条件流)
    Gate = missing_vars (B, C) 二值掩码
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, z_obs: torch.Tensor, z_era5: torch.Tensor,
                missing_vars: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_obs       : (B, C, d_model)  AWS 观测流编码
            z_era5      : (B, C, d_model)  ERA5 条件流编码
            missing_vars: (B, C)           1=该变量有缺失时间步
        Returns:
            z_fused     : (B, C, d_model)
        """
        # 交叉注意力：AWS 作为 Q，ERA5 作为 K/V
        cross_out, _ = self.cross_attn(
            query=z_obs, key=z_era5, value=z_era5)  # (B, C, d_model)

        # 变量级门控：只有有缺失的变量才接受 ERA5 信息
        gate = missing_vars.unsqueeze(-1).float()    # (B, C, 1)
        z_fused = z_obs + gate * self.drop(cross_out)
        return self.norm(z_fused)


class ECBIT(nn.Module):
    """
    ERA5-Conditioned Block Imputation Transformer (ECBIT).

    Architecture:
      1. Observation stream: iTransformer encoder on (X_obs || M_obs || T_enc)
      2. ERA5 condition stream: iTransformer encoder on (E || T_enc)
      3. Conditional cross-attention: variable-level gating
      4. Reconstruction head: per-variable linear projection → (B, T, C)

    Args:
        seq_len    : input/output sequence length T
        n_vars     : number of meteorological variables C (AntAWS = 5)
        n_time     : number of time encoding features (default 4)
        d_model    : hidden dimension (default 128)
        n_heads    : attention heads (default 8)
        n_layers   : encoder layers per stream (default 3)
        d_ff       : FFN dimension (default 256)
        use_era5   : enable ERA5 conditioning (ablation switch)
        use_cross  : enable cross-attention (ablation switch, requires use_era5)
        use_block_mask : use block missing simulation during training
    """
    def __init__(self, seq_len: int = 168, n_vars: int = 5, n_time: int = 4,
                 d_model: int = 128, n_heads: int = 8, n_layers: int = 3,
                 d_ff: int = 256, use_era5: bool = True,
                 use_cross: bool = True, dropout: float = 0.1):
        super().__init__()
        self.seq_len  = seq_len
        self.n_vars   = n_vars
        self.use_era5 = use_era5
        self.use_cross = use_cross

        # 观测流输入维度：n_vars 数值 + n_vars 可观测性掩码 + n_time 时间编码
        # 按变量分离：每个 variate token 的输入 = T 步的 (value, obs_mask, t_enc)
        obs_in_dim = 1 + 1 + n_time  # per-variable: value + obs_mask + time_enc

        self.obs_proj = nn.Linear(seq_len * obs_in_dim, d_model)
        self.obs_pos  = nn.Parameter(torch.randn(1, n_vars, d_model) * 0.02)
        obs_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout, batch_first=True, norm_first=True)
        self.obs_encoder = nn.TransformerEncoder(obs_layer, num_layers=n_layers)

        if use_era5:
            era5_in_dim = 1 + n_time  # per-variable: ERA5 value + time_enc
            self.era5_proj = nn.Linear(seq_len * era5_in_dim, d_model)
            self.era5_pos  = nn.Parameter(torch.randn(1, n_vars, d_model) * 0.02)
            era5_layer = nn.TransformerEncoderLayer(
                d_model, n_heads, d_ff, dropout, batch_first=True, norm_first=True)
            self.era5_encoder = nn.TransformerEncoder(era5_layer, num_layers=n_layers)

            if use_cross:
                self.cond_attn = ConditionalCrossAttention(d_model, n_heads, dropout)

        self.norm = nn.LayerNorm(d_model)
        # 重建头：每个变量独立投影回 T 步
        self.recon_head = nn.Linear(d_model, seq_len)

    def forward(self, x_obs: torch.Tensor, m_miss: torch.Tensor,
                e: torch.Tensor, t_enc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_obs  : (B, T, C)  观测值（缺失处已填零）
            m_miss : (B, T, C)  缺失掩码（1=缺失）
            e      : (B, T, C)  ERA5 序列（完整）
            t_enc  : (B, T, 4)  时间编码
        Returns:
            x_hat  : (B, T, C)  重建序列（全时间步）
        """
        B, T, C = x_obs.shape

        # 构建观测流输入：per-variable (value, obs_mask, time_enc)
        obs_mask = 1.0 - m_miss  # 1=可观测
        # 拼接后按变量分离：(B, T, C*(1+1+n_time)) → reshape → (B, C, T*(1+1+n_time))
        obs_input = torch.cat([x_obs, obs_mask, t_enc.unsqueeze(2).expand(B,T,C,4).reshape(B,T,C*4)], dim=-1)
        # 简化：直接用 x_obs + obs_mask 拼接，time_enc 广播
        x_aug = torch.cat([x_obs, obs_mask], dim=-1)  # (B, T, 2C)
        # 按变量分离：(B, T, 2C) → (B, C, T*2) for projection
        x_var = x_aug.transpose(1, 2).reshape(B, C, T * 2)  # (B, C, T*2)
        # 加入时间编码（广播到每个变量）
        t_var = t_enc.mean(dim=-1, keepdim=True).expand(B, T, C).transpose(1,2)  # (B,C,T)
        x_var = torch.cat([x_var, t_var], dim=-1)  # (B, C, T*2+T)

        z_obs = self.obs_proj(x_var) + self.obs_pos  # (B, C, d_model)
        z_obs = self.obs_encoder(z_obs)               # (B, C, d_model)

        # ERA5 条件流
        if self.use_era5:
            e_var = e.transpose(1, 2)                 # (B, C, T)
            e_var = torch.cat([e_var, t_var], dim=-1) # (B, C, T+T)
            z_era5 = self.era5_proj(e_var) + self.era5_pos
            z_era5 = self.era5_encoder(z_era5)        # (B, C, d_model)

            if self.use_cross:
                # 变量级缺失指示：该变量在窗口内是否有任何缺失时间步
                missing_vars = (m_miss.sum(dim=1) > 0).float()  # (B, C)
                z_fused = self.cond_attn(z_obs, z_era5, missing_vars)
            else:
                # 消融：简单相加（无门控）
                z_fused = z_obs + z_era5
        else:
            z_fused = z_obs

        z_fused = self.norm(z_fused)                  # (B, C, d_model)
        x_hat = self.recon_head(z_fused)              # (B, C, T)
        return x_hat.transpose(1, 2)                  # (B, T, C)
```

### 消融变体配置

| 变体名 | `use_era5` | `use_cross` | 说明 |
|--------|-----------|-------------|------|
| `ecbit_full` | ✓ | ✓ | 完整模型（条件化交叉注意力） |
| `ecbit_no_cross` | ✓ | ✗ | ERA5 简单相加，无门控 |
| `ecbit_no_era5` | ✗ | ✗ | 纯 AWS 观测流，无 ERA5 |
| `ecbit_no_blockmask` | ✓ | ✓ | 完整模型但训练时用随机点缺失 |

---

## Part 4 · 实验配置

### 超参数

| 参数 | 值 |
|------|-----|
| 序列长度 $T$ | 168 步（约 3 周，3h 分辨率） |
| 变量数 $C$ | 5（T, RH, wspd, P, q） |
| $d_{\text{model}}$ | 128 |
| $n_{\text{layers}}$ | 3（每流） |
| $n_{\text{heads}}$ | 8 |
| $d_{\text{ff}}$ | 256 |
| Batch size | 32 |
| 优化器 | AdamW（lr=$10^{-4}$，wd=$10^{-4}$） |
| 调度器 | 余弦退火 |
| Early Stopping | patience=10 |
| 损失函数 | MSE（仅缺失位置） |
| 随机种子 | 42 / 43 / 44 |

### 缺失模式设置

| 模式名 | 块长度范围 | 对应场景 |
|--------|-----------|---------|
| `short` | 6–24 步（18h–3天） | 传感器短暂结冰 |
| `medium` | 24–72 步（3–9天） | 电源故障 |
| `long` | 72–240 步（9–30天） | 冬季维护间隔 |

缺失率：20% / 40% / 60%

### 站点分组

| 组别 | 筛选标准 | 用途 |
|------|---------|------|
| 主实验站（~35个） | T 完整率 > 70%，记录 ≥ 10 年 | 训练+评估 |
| 高缺失子组 | WS 完整率 < 40% | 分析 ERA5 增益 |
| 低缺失子组 | WS 完整率 > 70% | 对照分析 |
| 泛化站（5个） | 地理多样性，未参与训练 | 泛化测试 |

---

## Part 5 · 4周精确执行计划

### Week 1（Days 1–7）· 数据工程

```
Day 1 — Initialization Session
  Hermes 调用 Claude Code:
  → 创建项目目录，写入 AGENTS.md / CLAUDE.md / README.md
  → git init，首次 commit
  → 验证所有 SKILLs，测试 microclimate-experiment-server
  → clawbot send "ECBIT Day 1: initialized ✓"

Day 2 — Station Selection Session
  Hermes 调用 Claude Code:
  → 实现 AntAWS 站点筛选脚本（T完整率>70%，≥10年记录）
  → 生成 data/station_meta_ecbit.csv（主实验站 + 泛化站标注）
  → commit: "[data] feat: station selection, N stations retained"

Day 3 — ERA5 Resampling Session
  Hermes 调用 Claude Code:
  → 实现 src/resample_era5.py（2h → 3h，与 AntAWS 时间戳对齐）
  → 本地 darts 运行，生成 data/era5_3h/
  → 单元测试：时间戳对齐验证，无 NaN
  → commit: "[data] feat: ERA5 resampled to 3h, aligned with AntAWS"

Day 4 — Missing Pattern Analysis Session
  Hermes 调用 Claude Code:
  → 实现 scripts/analyze_missing_patterns.py（本地 darts）
  → 统计：每站点每变量的块长度分布（mean/max/count）
  → 生成 results/missing_analysis/pattern_stats.csv
  → 写入 findings.md：AntAWS 真实块缺失统计（支撑论文动机）
  → commit: "[data] analysis: missing pattern statistics complete"

Day 5 — Preprocessing Pipeline Session
  Hermes 调用 Claude Code:
  → 实现 src/preprocess_antaws_impute.py
  → 实现 src/utils/block_missing.py（含单元测试）
  → 生成 data/processed/ .npz 文件（所有主实验站）
  → commit: "[data] feat: imputation preprocessing pipeline complete"

Day 6–7 — Buffer + Week 1 Review
  Hermes 调用 Claude Code:
  → 修复所有 bug，确认 Phase 1 所有任务 [x]
  → clawbot send "ECBIT Week 1 complete: data pipeline ready ✓"
```

### Week 2（Days 8–14）· 模型实现

```
Day 8 — ECBIT Core Model Session
  Hermes 调用 Claude Code:
  → 实现 src/models/ecbit.py（完整 ECBIT，含消融开关）
  → 单元测试（本地 darts）：
    - 前向传播输出形状 (B, T, C) ✓
    - use_era5=False 退化为纯 AWS 流 ✓
    - 变量级门控：可观测变量不受 ERA5 影响 ✓
    - 损失只在缺失位置计算 ✓
  → commit: "[model] feat: ECBIT full model with ablation switches"

Day 9 — Baselines Session
  Hermes 调用 Claude Code:
  → 实现 src/baselines/linear_interp.py, locf.py
  → 实现 src/baselines/saits_wrapper.py（PyPOTS SAITS）
  → 实现 src/baselines/brits_wrapper.py（PyPOTS BRITS）
  → 实现 src/baselines/itransformer_impute.py
  → 实现 src/baselines/era5_direct.py（训练集偏差校正后直接替代）
  → 所有基线单元测试通过
  → commit: "[model] feat: all 6 baseline models implemented"

Day 10 — Training Framework Session
  Hermes 调用 Claude Code:
  → 实现 src/train_impute.py（统一训练入口，YAML config，W&B 日志）
  → 实现 src/evaluate_impute.py（per-variable MAE/RMSE + mean）
  → 生成所有实验 YAML 配置文件（round1/2/3）
  → commit: "[infra] feat: training framework + all experiment configs"

Day 11 — Smoke Test Session
  Hermes 调用 Claude Code:
  → 提交 smoke test job（ECBIT full，1站点，medium missing，rate=0.3，seed=42，10 epochs）
  → 验证：result JSON 存在，MAE 非 NaN，W&B run 出现
  → commit: "[infra] test: smoke test passed on remote server"

Day 12–14 — Buffer + Pre-Experiment Check
  Hermes 调用 Claude Code:
  → 修复所有 bug，确认 Phase 2 所有任务 [x]
  → 准备 paper/main.tex（TGRS 模板，章节框架，references.bib 预填充）
  → clawbot send "ECBIT Week 2 complete: model + infra ready ✓"
```

### Week 3（Days 15–21）· 实验执行 ‖ 论文 Introduction + Related Work + Methodology

```
Day 15 — Round 1 Submission + Introduction Session
  Hermes 调用 Claude Code:
  ① 提交 Round 1 所有 162 个基线 job
     → clawbot send "ECBIT Round 1 submitted: 162 baseline jobs running"
  ② [立即切换] 写 paper/sections/introduction.tex
     → AntAWS 缺失问题统计（引用 findings.md 数据）
     → 块缺失 vs 随机点缺失的本质差异
     → ERA5 条件化的动机（来自 ECAFT 发现）
     → 贡献声明（3条）
     → commit: "[paper] write: Introduction section"

Day 16 — Round 1 Monitoring + Related Work Session
  Hermes 调用 Claude Code:
  ① 轮询 Round 1 状态，拉取已完成结果
  ② [并行] 使用 autoresearch SKILL 补充文献调研
     → 写 paper/sections/related_work.tex
     → 覆盖：SAITS/BRITS/CSDI imputation，极地 AWS 数据，ERA5 偏差校正
     → commit: "[paper] write: Related Work section"

Day 17 — Round 1 Results + Methodology §3.1–3.2 Session
  Hermes 调用 Claude Code:
  ① 拉取所有 Round 1 结果，生成 baseline_table.csv
     → clawbot send "ECBIT Round 1 done. Best baseline: <model> MAE=X.XX"
  ② [并行] 写 methodology.tex §3.1–3.2
     → Problem formulation（符号定义，X/M/E/T_enc）
     → Block missing simulation（公式 + 与 MCAR 对比）
     → commit: "[paper] write: Methodology 3.1-3.2"

Day 18 — Round 2 Submission + Methodology §3.3–3.5 Session
  Hermes 调用 Claude Code:
  ① 提交 Round 2 所有 108 个消融 job
     → clawbot send "ECBIT Round 2 submitted: 108 ablation jobs running"
  ② [立即切换] 写 methodology.tex §3.3–3.5
     → Observation stream encoder（iTransformer variate-token）
     → ERA5 condition stream
     → Conditional cross-attention（变量级门控，公式）
     → Training objective（仅缺失位置 MSE）
     → commit: "[paper] write: Methodology 3.3-3.5"

Day 19 — Round 2 Monitoring + TikZ Architecture Diagram Session
  Hermes 调用 Claude Code:
  ① 轮询 Round 2 状态，拉取已完成结果
  ② [并行] 使用 academic-plotting SKILL 生成 TikZ 架构图
     → paper/figures/fig1_ecbit_arch.tex
     → 展示：双流输入 → 两个 iTransformer Encoder → 条件化交叉注意力 → 重建头
     → 标注变量级门控机制（missing_vars 掩码）
     → commit: "[paper] fig: TikZ architecture diagram (Fig.1)"

Day 20 — Round 2 Results + Round 3 Submission Session
  Hermes 调用 Claude Code:
  ① 拉取所有 Round 2 结果，生成 ablation_table.csv
     → 关键验证：ecbit_full < ecbit_no_era5 on long-block missing
     → clawbot send "ECBIT Round 2 done. Full vs no-ERA5: X.XX vs X.XX"
  ② 提交 Round 3 泛化测试 45 个 job
     → clawbot send "ECBIT Round 3 submitted: 45 generalization jobs"

Day 21 — Buffer + Week 3 Review
  Hermes 调用 Claude Code:
  → 修复实验问题，确认 Phase 3 实验任务全部 [x]
  → clawbot send "ECBIT Week 3 complete: all experiments done, Intro+Related+Method written ✓"
```

### Week 4（Days 22–28）· 可视化 + Experiments 撰写 + 精修投稿

```
Day 22 — Visualization Session (Fig.2–5)
  Hermes 调用 Claude Code:
  → 拉取 Round 3 结果，生成 generalization_table.csv
  → 实现并运行 scripts/plot_missing_patterns.py → Fig.2
  → 实现并运行 scripts/plot_main_results.py → Fig.3
  → 实现并运行 scripts/plot_era5_effect.py → Fig.4
  → 实现并运行 scripts/plot_imputation_curves.py → Fig.5
  → commit: "[paper] fig: all 4 data figures generated"

Day 23 — Experiments §4.1–4.3 Session
  Hermes 调用 Claude Code:
  → 写 experiments.tex §4.1–4.2（Setup + Main Results Table I）
  → 写 experiments.tex §4.3（Ablation Table II）
  → 所有数值从 CSV 文件读取，不手动输入
  → commit: "[paper] write: Experiments 4.1-4.3 + Table I/II"

Day 24 — Experiments §4.4–4.5 + Conclusion Session
  Hermes 调用 Claude Code:
  → 写 experiments.tex §4.4（ERA5 conditioning analysis，引用 Fig.4）
  → 写 experiments.tex §4.5（Generalization Table III）
  → 写 conclusion.tex（总结，局限性，future work）
  → 写 Abstract（≤250词）
  → commit: "[paper] write: Experiments 4.4-4.5 + Conclusion + Abstract"

Day 25 — Full Paper Review Session
  Hermes 调用 Claude Code:
  → 使用 ml-paper-writing SKILL citation verification workflow
  → latexmk 编译完整 PDF，检查：
    - 所有图表正常显示，引用编号正确
    - 页数 ≤ 10 页（TGRS 要求）
    - 无未定义引用
  → 标注需修改处，写入 findings.md
  → commit: "[paper] review: full draft compiled, issues logged"

Day 26 — Polish Session
  Hermes 调用 Claude Code:
  → 精修 Introduction + Related Work（措辞、引用格式）
  → 精修 Methodology（公式符号统一，图表引用）
  → 精修 Experiments（数值核对，表格格式）
  → commit: "[paper] polish: full paper polished"

Day 27 — Format Check + Final PDF Session
  Hermes 调用 Claude Code:
  → IEEE TGRS 格式合规检查（双栏，字体，页数，参考文献格式）
  → latexmk -pdf 生成最终 PDF
  → Adobe Acrobat 检查字体嵌入
  → commit: "[paper] final: submission-ready PDF"
  → clawbot send "ECBIT paper ready for submission ✓"

Day 28 — Submission
  Hermes 调用 Claude Code:
  → 提交至 IEEE TGRS 或 Remote Sensing
  → 更新 task_plan.md：所有任务 [x]
  → commit: "[paper] submit: submitted"
  → clawbot send "🎉 ECBIT submitted!"
```

---

## Part 6 · 异常处理

| 异常 | 判断标准 | 处理 |
|------|---------|------|
| 服务器 OOM | log 含 `CUDA out of memory` | 减半 batch_size，重提交；clawbot 通知 |
| ERA5 重采样失败 | 时间戳不对齐 | 检查 AntAWS 时区（UTC），重新对齐；clawbot 通知 |
| SAITS/BRITS 安装失败 | PyPOTS import error | `pip install pypots`，检查版本兼容性 |
| 消融结果无差异 | ecbit_full ≈ ecbit_no_era5 | 分析长块缺失子集（>72步），ERA5 增益可能只在极长缺失时显著；调整论文叙事 |
| 本地内存过高 | 进程占用 >80% RAM | `scripts/release_resources.sh`；clawbot 通知 |

```bash
# ClawBot 通知模板
clawbot send "✅ ECBIT [Phase]: <task> complete. Next: <next_task>"
clawbot send "📊 ECBIT Round <N> done. ECBIT MAE=<X> vs best_baseline=<Y>"
clawbot send "🚨 ECBIT BLOCKED: <error>. Action: <action>. Check needed."
clawbot send "🎉 ECBIT submitted to <venue>!"
```

---

## Part 7 · 投稿检查清单

```
格式合规（IEEE TGRS）
- [ ] 双栏，Times New Roman 10pt
- [ ] 页数 ≤ 10 页（含参考文献）
- [ ] 所有图表字体 ≥ 8pt，分辨率 ≥ 300 DPI
- [ ] 参考文献 IEEE 格式

内容完整
- [ ] Abstract ≤ 250 词，含数据集/方法名/主要指标
- [ ] 所有实验数值与 results/tables/ CSV 一致
- [ ] Fig.1–5 和 Table I–III 均在正文中有引用
- [ ] ECAFT 失败发现在 Introduction 中作为动机引用

技术检查
- [ ] latexmk 编译无错误
- [ ] PDF 字体全部嵌入
- [ ] W&B 实验记录完整（315 runs）
- [ ] 代码仓库整理，README 含复现指引

提交操作
- [ ] IEEE TGRS 投稿系统账号注册
- [ ] 作者信息、关键词填写
- [ ] PDF 上传，系统确认接收
- [ ] clawbot send "🎉 ECBIT submitted!"
```
