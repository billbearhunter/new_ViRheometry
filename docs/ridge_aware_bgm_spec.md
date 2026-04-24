# Ridge-Aware BGM 完整方案数学规范(v2 修订版)

## 修订历史

- **v1**(2026-04-23):初始 spec,提出 joint $(φ, φ^{PP})$ BGM,声称 ridge invariance theorem 和 box coverage theorem。归档于 [`ridge_aware_bgm_spec_v1_archived.md`](ridge_aware_bgm_spec_v1_archived.md)。
- **v2**(2026-04-24,本文档):基于多个尖锐质疑修订:
  - φ^PP 弱化为"经 Hamamichi-PP 近似的 ridge proxy",非严格不变量
  - Theorem 1 改为 Approximation Lemma
  - Theorem 2 弱化为 Theorem 2'(条件 box coverage,依赖训练支撑)
  - 新增 §3:边缘化后 cluster 在 φ 空间的多模性问题
  - 新增 §4:rBCM posterior weight 缺失 bug(应先于 ridge-aware 修)
  - 新增 §5:验证-优先实施路径(原 v1 的"立刻重训"被拒)
  - 新增 §6:评价指标改为 top-M coverage,而非 top-1 accuracy
  - 新增 §7:数据范围扩展(v3 + 4177 行 infill 合成 v4)

---

## 1. Problem Setup

**Inverse rheometry**:给定一组 dam-break 观测 $\mathcal{D}_{\text{obs}} = \{(W_\kappa, H_\kappa, \mathbf{y}_\kappa)\}_{\kappa=1}^{K_s}$ ($K_s$ 个 setup),恢复 Herschel-Bulkley 参数 $\boldsymbol{\theta} = (n, \eta, \sigma_y) \in \Theta \subset \mathbb{R}^3$。

**Forward MPM**:$\mathbf{y} = \mathcal{F}_{\text{MPM}}(\boldsymbol{\theta}; W, H) \in \mathbb{R}^8$(8 frame flow distances)。

**目标**:替代 $\mathcal{F}_{\text{MPM}}$ 用 surrogate $\hat{\mathcal{F}}_{\text{rBCM}}$ 实现 real-time inverse,同时保持精度。

---

## 2. Surrogate 架构(2-Level Hierarchical Mixture-of-Experts)

### 2.1 因果分解

$\boldsymbol{\theta}, W, H$ 为输入,$\mathbf{y}$ 为响应。Hierarchical 路由因果地分解为:

$$
p(g, k \mid W, H, \mathbf{y}, \boldsymbol{\theta}) \;=\; \underbrace{p(g \mid W, H)}_{\text{geometry partition}} \cdot \underbrace{p(k \mid \mathbf{y}, W, H, g; \boldsymbol{\theta})}_{\text{response partition (ridge-aware)}}
$$

**Stage 1** 仅依赖 $(W, H)$:实验设置是 exogenous,routing 不应依赖材料。
**Stage 2** 依赖 $\mathbf{y}$ 和(训练时)$\boldsymbol{\theta}$ 通过 ridge identity proxy 引入。

### 2.2 Stage 1 — Geometry BGM(不变)

#### Feature

$\mathbf{x}_i^{(1)} = (W_i, H_i) \in \mathbb{R}^2$。

#### Generative model

$$
p^{(1)}(\mathbf{x}^{(1)}) = \sum_{g=1}^{K_{\text{geo}}} \pi_g^{(1)}\, \mathcal{N}\!\bigl(\mathbf{x}^{(1)}; \boldsymbol{\mu}_g^{(1)}, \boldsymbol{\Sigma}_g^{(1)}\bigr)
$$

#### BIC

$$
\boxed{\text{BIC}^{(1)}(K_{\text{geo}}) = -2\log \mathcal{L}^{(1)} + (6K_{\text{geo}} - 1)\log N}
$$

每成分 $2$(均值)$+ 3$(对称协方差)$+ 1$(权重 $-1$ 归一化)= $6$ 个有效参数,加 $K-1$ 混合权重约束 → $6K - 1$。扫描 $K_{\text{geo}} \in \{2, ..., 15\}$,选最小 BIC 的 $K_{\text{geo}}^* = 12$。

#### Routing

$$
g^*(W, H) = \arg\max_{g} \pi_g^{(1)}\, \mathcal{N}\!\bigl((W, H); \boldsymbol{\mu}_g^{(1)}, \boldsymbol{\Sigma}_g^{(1)}\bigr)
$$

#### Scope caveat

Stage 1 只用 $(W, H)$ 的前提**仅在我们当前 dam-break 实验只变 $(W, H)$ 时成立**。若将来加入其他 setup descriptor(闸门提升速度、容器表面粗糙度、起始倾斜等),Stage 1 feature 应当扩展。

> **Lemma 1**(Geometry partition invariance — 设计选择):若 $\mathbf{y}$ 进入 Stage 1 feature,则同一 $(W_0, H_0)$ 在不同材料 $\boldsymbol{\theta}_1, \boldsymbol{\theta}_2$ 下产生 $\mathbf{y}_1 \ne \mathbf{y}_2$ 会被路由到不同 geo group。这违反 "geometry 是 exogenous 不变量" 的物理直觉。Lemma 1 是**因果建模选择**,不是数学定理。

### 2.3 Stage 2 — Ridge-aware Response BGM(本工作的新贡献,带重要 caveat)

#### Hamamichi ridge invariant — **是 proxy 不是 invariant**

定义 plane-Poiseuille 投影([Hamamichi et al. 2023, Algorithm 2]):

$$
\boldsymbol{\phi}^{PP}: (\boldsymbol{\theta}, W, H) \mapsto (P_0, L_0) \in \mathbb{R}^2
$$

由 [`mat_hw_to_PL`](Optimization/libs/compare_loss.py) 实现(66 系数 order-2 多项式)。

#### Approximation Lemma(取代原 Theorem 1)

> **Lemma A1**(φ^PP as approximate ridge proxy):**Hamamichi 论文 §5.3 footnote 15 自承** "$H^{PP}$ 不是 $H$ 的精确近似,condition number 不严格匹配"。所以:
> - **正向**:在 PP 模型适用域(中等 σ_y、远离 n→1 的 shear-thinning)$\theta_a \sim_{\text{ridge}} \theta_b \implies \|\phi^{PP}_a - \phi^{PP}_b\| < \epsilon$
> - **反向不严格**:$\phi^{PP}$ 接近不严格保证 ridge-equivalent。可能有"伪同 ridge"的样本被错误聚合。
> - **边界失效区**:极小 σ_y(< 0.01)、近牛顿(n > 0.95)、薄材料(η < 1):$\phi^{PP}$ 投影本身可能 ill-conditioned。

#### 必需的经验验证(原 v1 缺失,见 §5)

φ^PP 在 v3 数据上的 ridge proxy 质量必须经验测量后才能信任。验证方法:在训练数据中找 $y$ 距离极近的样本对,测 $\|\phi^{PP}_i - \phi^{PP}_j\|$ 的分布。

#### Feature

$\mathbf{x}_i^{(2,g)} = (\boldsymbol{\phi}_i, \boldsymbol{\phi}^{PP}_i) \in \mathbb{R}^{18 + 2} = \mathbb{R}^{20}$,其中 $i$ 限制在 $g^*(W_i, H_i) = g$ 子集内。

| 块 | 维数 | 含义 | 推理可计算? |
|---|---|---|---|
| $\boldsymbol{\phi}_i \in \mathbb{R}^{18}$ | y-shape feature | $\bigl[\mathbf{y}_i/y_{i,8},\ \Delta(\mathbf{y}_i/y_{i,8}),\ \log y_{i,8},\ \log\sqrt{W_iH_i},\ \log(W_i/H_i)\bigr]$ | **是** |
| $\boldsymbol{\phi}^{PP}_i \in \mathbb{R}^{2}$ | (proxy) ridge identity | $\text{mat\_hw\_to\_PL}(\eta_i, n_i, \sigma_{y,i}, W_i, H_i)$ | **否(需 $\boldsymbol{\theta}$)** |

#### Generative model(条件独立块对角)

$$
p^{(2,g)}(\boldsymbol{\phi}, \boldsymbol{\phi}^{PP}) = \sum_{k=1}^{K_\phi^{(g)}} \pi_k^{(g)}\, \underbrace{\mathcal{N}\!\bigl(\boldsymbol{\phi}; \boldsymbol{\mu}_k^{\phi,(g)}, \boldsymbol{\Sigma}_k^{\phi,(g)}\bigr)}_{p_k^\phi(\boldsymbol{\phi})}\, \underbrace{\mathcal{N}\!\bigl(\boldsymbol{\phi}^{PP}; \boldsymbol{\mu}_k^{PP,(g)}, \boldsymbol{\Sigma}_k^{PP,(g)}\bigr)}_{p_k^{PP}(\boldsymbol{\phi}^{PP})}
$$

**条件独立假设**:给定 cluster $k$,$\boldsymbol{\phi}$ 与 $\boldsymbol{\phi}^{PP}$ 块对角独立。

> **Justification**:cluster 是 "y-shape × (proxy)ridge-identity" 的联合 mode。物理上,同 cluster 的样本应同时具有相似的观测形状 AND 同一(近似)similarity ridge。块对角的统计弱化避免了 over-parameterization。

#### EM Updates

**E-step**:
$$
\gamma_{ik} = \frac{\pi_k^{(g)}\, p_k^\phi(\boldsymbol{\phi}_i)\, p_k^{PP}(\boldsymbol{\phi}^{PP}_i)}{\sum_{k'} \pi_{k'}^{(g)}\, p_{k'}^\phi(\boldsymbol{\phi}_i)\, p_{k'}^{PP}(\boldsymbol{\phi}^{PP}_i)}
$$

**M-step**(块对角让 $\phi$ 和 $\phi^{PP}$ 块独立更新):
$$
N_k = \textstyle\sum_i \gamma_{ik},\qquad \pi_k^{(g)} = N_k / N_g
$$
$$
\boldsymbol{\mu}_k^\phi = \tfrac{1}{N_k}\textstyle\sum_i \gamma_{ik}\, \boldsymbol{\phi}_i,\quad
\boldsymbol{\Sigma}_k^\phi = \tfrac{1}{N_k}\textstyle\sum_i \gamma_{ik}(\boldsymbol{\phi}_i - \boldsymbol{\mu}_k^\phi)(\boldsymbol{\phi}_i - \boldsymbol{\mu}_k^\phi)^\top + \delta I
$$
$$
\boldsymbol{\mu}_k^{PP} = \tfrac{1}{N_k}\textstyle\sum_i \gamma_{ik}\, \boldsymbol{\phi}^{PP}_i,\quad
\boldsymbol{\Sigma}_k^{PP} = \tfrac{1}{N_k}\textstyle\sum_i \gamma_{ik}(\boldsymbol{\phi}^{PP}_i - \boldsymbol{\mu}_k^{PP})(\boldsymbol{\phi}^{PP}_i - \boldsymbol{\mu}_k^{PP})^\top + \delta I
$$

$\delta I$ 是数值正则($\delta = 10^{-5}$)。收敛准则:相对 log-likelihood 变化 $< 10^{-4}$ 或 max_iter = 300。

#### BIC

每成分参数:
| 块 | 个数 |
|---|---|
| $\boldsymbol{\mu}_k^\phi$ | 18 |
| $\boldsymbol{\Sigma}_k^\phi$(对称) | $18 \cdot 19 / 2 = 171$ |
| $\boldsymbol{\mu}_k^{PP}$ | 2 |
| $\boldsymbol{\Sigma}_k^{PP}$(对称) | 3 |
| **小计** | **194** |

$$
\boxed{\text{BIC}^{(2, g)}(K_\phi^{(g)}) = -2\log \mathcal{L}^{(2, g)} + (195 K_\phi^{(g)} - 1)\log N_g}
$$

扫描 $K_\phi^{(g)} \in \{8, 10, 12, 15, 20, 25, 30, 40, 50\}$ per geo,选最小 BIC。

> **重要 caveat**:**BIC 最小 ≠ inverse 精度最优**。BIC 测的是 joint $(\phi, \phi^{PP})$ 密度拟合,不直接对应真值恢复。BIC-optimum K 只作 starting point,**最终 $K_\phi$ 应根据 5 真实材料的 inverse RMSE 在 BIC top-3 候选中再选**。

#### Inference-time Routing

推理时 $\boldsymbol{\theta}$ 未知,$\boldsymbol{\phi}^{PP}$ 不可计算。**对 $\boldsymbol{\phi}^{PP}$ 边缘化**:

$$
p(k \mid \boldsymbol{\phi}, g) = \int p(k, \boldsymbol{\phi}^{PP} \mid \boldsymbol{\phi}, g)\, d\boldsymbol{\phi}^{PP} = \frac{\pi_k^{(g)}\, p_k^\phi(\boldsymbol{\phi})}{\sum_{k'} \pi_{k'}^{(g)}\, p_{k'}^\phi(\boldsymbol{\phi})}
$$

$$
\boxed{k^*(\boldsymbol{\phi}, g) = \arg\max_k \pi_k^{(g)}\, \mathcal{N}\!\bigl(\boldsymbol{\phi}; \boldsymbol{\mu}_k^{\phi,(g)}, \boldsymbol{\Sigma}_k^{\phi,(g)}\bigr)}
$$

**形式同 v2**,但 $(\boldsymbol{\mu}_k^\phi, \boldsymbol{\Sigma}_k^\phi)$ 来自联合训练,数值不同。

> **关键缺陷:边缘化后 cluster 在 φ 空间可能仍重叠**。即便 $\mu^{PP}_A \ne \mu^{PP}_B$(不同 ridge),若 $\mu^\phi_A \approx \mu^\phi_B$(y-shape 接近),边缘 $p(k | \phi)$ 在 A 和 B 上几乎平均 → gate 仍 confused。
>
> **必须 dry-run 验证**:训完 BGM 后,对每 cluster pair 算 marginal $p(A|\phi)$ vs $p(B|\phi)$ 在交集区域的 KL 散度;若多 pair KL ≈ 0,ridge-aware 在边缘化后失效,需要切换到 multi-modal posterior routing(见 §6)。

### 2.4 Box Coverage(弱化版)

> **Theorem 2'**(Conditional ridge coverage,弱化版):令 $C_k^{(g)}$ 为 cluster $k$ 在 geo $g$ 中的训练样本集。在 Ridge-BGM 收敛后:
> $$\forall i, j \in C_k^{(g)}: \;\; \|\boldsymbol{\phi}^{PP}_i - \boldsymbol{\phi}^{PP}_j\| < \epsilon_k$$
> 其中 $\epsilon_k = \mathcal{O}(\sqrt{\text{tr}(\boldsymbol{\Sigma}_k^{PP})})$。即同 cluster 的训练样本**在 φ^PP proxy 意义下近似同 ridge**(受 Lemma A1 限制)。
>
> **Cluster 训练 box 在 ridge 方向上的 extent = 训练样本沿 ridge 的 max-min**。**Ridge-aware 不创造数据**——只把现有数据沿 ridge 方向"集中"到正确 cluster。

> **Corollary(条件)**:对推理时真值 $\boldsymbol{\theta}^*$,routing 选 $k^*$。若**且仅若** $\exists \boldsymbol{\theta}_i \in C_{k^*}$ 满足:
> 1. $\|\phi^{PP}(\boldsymbol{\theta}^*, W, H) - \phi^{PP}(\boldsymbol{\theta}_i, W_i, H_i)\| < \epsilon_{k^*}$(同 ridge proxy)
> 2. $\boldsymbol{\theta}^*$ 与 $\boldsymbol{\theta}_i$ 之间 ridge 段无 box 中断
>
> 则 cluster box 覆盖 $\boldsymbol{\theta}^*$。**两个条件都不自动成立**。

> **Limitation 明确**:
> - 若真值 ridge **完全没有训练样本**(纯 data gap),无可救——需 target-material infill,正交于本方案
> - 若 φ^PP 在真值附近 ill-conditioned(Lemma A1 边界区),ridge proxy 失效
> - 若 cluster 训练样本沿 ridge 不连续,box 包络可能"跳过"真值

---

## 3. 边缘化 multimodality 问题(新增,原 v1 未讨论)

### 3.1 问题陈述

Ridge-aware BGM 在训练时通过 $\phi^{PP}$ 区分 cluster A 和 B(它们在不同 ridge 上)。但推理时只能看 $\phi$,marginal $p(k | \phi)$ 形式与 v2 相同。

**失效场景**:cluster A 和 B 的 $(\mu^\phi, \Sigma^\phi)$ 接近(因为 y-shape 在 PP-proxy 失效区可能不区分 ridge)。Marginal posterior $p(k | \phi)$ 是 bimodal(在 A 和 B 上都非零),hard top-1 routing 50/50 错。

### 3.2 验证指标

对 22k 测试集:
$$
H_{\text{posterior}}(\phi_i) = -\sum_k p(k | \phi_i) \log p(k | \phi_i)
$$
平均熵 $\bar H$。$\bar H$ 越小 → routing 越确信;$\bar H$ 接近 $\log K$ → 完全 confused。

对每 cluster pair $(A, B)$ 同 geo:
$$
D_{\text{KL}}(p_A^\phi \| p_B^\phi) = \int p_A^\phi(\phi) \log \frac{p_A^\phi(\phi)}{p_B^\phi(\phi)} d\phi
$$
若多 pair KL < 1,边缘化后 cluster 不可分。

### 3.3 应对策略

若 marginal multimodal 严重,**必须放弃 hard top-1**,改用:
- **Top-M routing**:返回 top-M cluster 候选(按 marginal posterior 降序)
- **Posterior-weighted rBCM**:聚合 M 个 expert,权重 = marginal posterior × GP precision(见 §4)

---

## 4. 先决修复:rBCM Posterior Weight Bug(新增,原 v1 漏)

### 4.1 当前 v2 的 bug

[`vi_mogp/model.py`](vi_mogp/model.py) 中 rBCM 的 $\beta_k$ 只来自 GP variance:
$$
\beta_k = \tfrac{1}{2}\bigl[\log \sigma_{\text{prior}}^2 - \log \sigma_k^2(\boldsymbol{\theta})\bigr]_+
$$

**即便 routing 给出 top-K candidates 和 BGM marginal posterior $p(k | \phi)$,rBCM 聚合完全忽略 $p(k | \phi)$**——只用 GP variance。BGM 的不确定性信息被丢弃。

### 4.2 修复

正确的 weight:
$$
\beta_k^{\text{corrected}} = p(k | \phi, g) \cdot \tfrac{1}{2}\bigl[\log \sigma_{\text{prior}}^2 - \log \sigma_k^2(\boldsymbol{\theta})\bigr]_+
$$

$p(k | \phi, g)$ 是 BGM marginal posterior(已知,gate 自然给出)。

### 4.3 必要性

- **修这个 bug 是 ridge-aware 重训的前提**——否则即便新 BGM 给出更好的 marginal posterior,聚合阶段也用不上
- **应该作为 baseline 改进 single-handedly 验证**:仅修 rBCM weight,不动 BGM,看能否在 5 真实材料上追上老方法。**如果可以,后面的 ridge-aware 重训都不必做**。

---

## 5. 验证-优先实施路径(取代原 v1 的"立刻重训")

### Step A:**修 rBCM posterior weight**(~2h dev,无需重训)

- 改 [`vi_mogp/model.py`](vi_mogp/model.py) 的 `predict_rbcm` / `predict_grbcm`,把 BGM posterior 传入并乘进 β
- 对 5 真实材料(Tonkatsu_2, Sweet_2 等)跑 inverse,比 flow-curve RMSE
- **Stop condition**:若 v2 + rBCM-fix 已追上老方法,**整个 ridge-aware 项目可暂缓**

### Step B:**φ^PP 经验验证**(~1h)

- 在 v3 训练数据中找样本对 $(i, j)$ 使 $\|y_i - y_j\| < \tau_y$ 且同 $(W, H)$
- 计算 $r_{ij} = \|\phi^{PP}_i - \phi^{PP}_j\| / \|\phi^{PP}_i\|$
- 中位数 $r$ < 0.1 → φ^PP 在 v3 上是良好 ridge proxy → 可继续 Step C
- 中位数 $r$ > 0.3 → φ^PP 在我们数据上失效 → **整个 ridge-aware 思路重审**

### Step C:**Ridge-aware BGM dry-run**(~3h,仅 BGM 不重训 GP)

合并 v3 + 4177 行 infill_clean → v4 训练集

- 预算 $\phi^{PP}$ 给 v4 的 ~300k 行(~30 s)
- 对每 geo 跑 ridge-aware EM,扫描 $K_\phi^{(g)}$(~30 min)
- 输出新 cluster 分配 + marginal $p(k|\phi)$ 参数
- **关键诊断**:
  1. 5 真实材料的 routed cluster box 是否包含真值?(boolean)
  2. Top-1 marginal 概率是否仍 ≈ 1.0?(高 = 仍 confident,可能仍错;低 = 自承歧义)
  3. Top-3 cluster 的 box 并集是否包含真值?
  4. 22k 测试集的 marginal posterior 平均熵 $\bar H$
  5. cluster pair 的 KL 散度分布(检查 §3.2 的 multimodal 问题)
- **Stop condition**:若新 routed box 仍不覆盖真值且 top-3 联合 box 也不覆盖 → ridge-aware 数学正确但实际无效,放弃,转 target-material infill

### Step D:**改评价指标**(无开发)

把 inverse benchmark 的 reporting 从单一 top-1 改为:
- Top-1 box-coverage rate(原指标)
- Top-3 box-coverage rate
- Posterior-weighted box coverage:$\sum_k p(k|\phi) \cdot \mathbb{1}[\theta^* \in \mathcal{B}_k]$
- Inverse RMSE on flow curve(物理)
- Inverse RMSE on parameters(参数空间)

### Step E:**全量重训**(~9h GPU,仅在 A-D 都通过时才做)

- 数据集 = v4 = v3 (295,419) + infill_clean (4,177) ≈ 299,596 行
- 用 Step C 选定的 $K_\phi^{(g)}$
- 训练 ~300-360 个 GP expert
- baseline 同 v2 流程

### Step F:**对比验证**

3 个变体:
1. v2 + rBCM weight fix
2. v3-ridge(新 BGM,v4 数据)+ rBCM weight fix
3. v3-ridge + top-M aggregation

5 真实材料 + 22k test set,看哪个真正改善 flow-curve RMSE 和 inverse 精度。

---

## 6. 评价指标(新增)

### 6.1 单 setup 不可辨识 → top-1 不是合理目标

Hamamichi similarity 定理:单 setup 信息论下限决定 1 个 ridge 维度无法恢复。**"唯一正确 cluster" 是 ill-defined goal**。

### 6.2 新指标体系

| 指标 | 定义 | 含义 |
|---|---|---|
| Top-1 BoxCov | $\mathbb{1}[\theta^* \in \mathcal{B}_{k^*}]$ | routing 正确性(严格)|
| Top-M BoxCov | $\mathbb{1}[\exists k \in \text{top-M},\ \theta^* \in \mathcal{B}_k]$ | M 候选中至少一个对 |
| Posterior-weighted BoxCov | $\sum_k p(k\|\phi) \cdot \mathbb{1}[\theta^* \in \mathcal{B}_k]$ | gate 总质量分配到正确 cluster 的比例 |
| Marginal entropy $\bar H$ | $-\sum_k p(k\|\phi) \log p(k\|\phi)$ | gate 的 confusion(高 = 自承歧义)|
| Flow-curve RMSE | $\|\sigma(\dot\gamma; \hat\theta) - \sigma(\dot\gamma; \theta^*)\|$ | inverse 物理精度(终极指标) |
| Parameter L2 | $\|\hat\theta - \theta^*\|_2$ | inverse 参数误差(似有相似性偏差,需谨慎)|

### 6.3 报告格式

不再追求"top-1 = 100%",改为:
- "Top-3 BoxCov = X%" — 在 v3-ridge 上应该 > v2
- "Flow-curve RMSE = Y cm" — 物理硬指标,与老方法直接比

---

## 7. 数据范围扩展:v3 + Infill → v4(新增)

### 7.1 数据来源

- **v3 base**:`TrainingData/moe_workspace_merged_v3_20260419/train_merged.csv`(295,419 行)
- **Hierarchical-BO infill**:`Optimization/hier_bo_infill_20260423_102604_full5000/infill_clean.csv`(4,177 行;1,076 AGT + 3,101 MLS)
- **合并产出**:`TrainingData/moe_workspace_merged_v4_20260424/train_merged.csv`(估 ~299,500 行,去重后)

### 7.2 合并工具

```bash
python -m DataPipeline.build_merged_dataset \
    --base-dir TrainingData/moe_workspace_merged_v3_20260419 \
    --infill Optimization/hier_bo_infill_20260423_102604_full5000/infill_clean.csv \
    --out-dir TrainingData/moe_workspace_merged_v4_20260424 \
    --dedup-precision 4
```

(注:val/test 保持 v3 不变,以保 before/after 评估可比。)

### 7.3 v4 数据的 φ^PP 预算

```bash
python -m DataPipeline.precompute_phi_pp \
    --train TrainingData/moe_workspace_merged_v4_20260424/train_merged.csv \
    --out Models/full_partition_v4/phi_pp_cache.npz
```

为 Step C 的 ridge-aware BGM 提供输入。

---

## 8. 每 Cluster 的 GP Expert(同 v2,不变)

每 cluster $(g, k)$ 对应一个 ExactGP expert,Matern-5/2 ARD on 5D 输入 $(n, \log \eta, \log \sigma_y, W, H)$。

预测:
$$
\hat{\mathbf{y}}_{(g,k)}(\boldsymbol{\theta}, W, H) = \boldsymbol{\mu}_{(g,k)}^{GP}(\boldsymbol{\theta}, W, H),\quad \hat{\boldsymbol{\sigma}}^2 = \boldsymbol{\Sigma}_{(g,k)}^{GP}
$$

---

## 9. rBCM 聚合(修订:加 BGM posterior weight)

**v2 原版**:$\beta_k = \tfrac{1}{2}[\log \sigma_{\text{prior}}^2 - \log \sigma_k^2(\boldsymbol{\theta})]_+$ — 漏了 BGM posterior

**v3-ridge 修订版**:
$$
\beta_k = p(k | \phi, g^*) \cdot \tfrac{1}{2}\bigl[\log \sigma_{\text{prior}}^2 - \log \sigma_k^2(\boldsymbol{\theta})\bigr]_+
$$
$$
\hat{\mathbf{y}}_{\text{rBCM}}(\boldsymbol{\theta}, W, H) = \sigma_{\text{rBCM}}^2 \sum_{k} \beta_k \sigma_k^{-2}\, \boldsymbol{\mu}_{(g^*, k)}^{GP},\quad
\sigma_{\text{rBCM}}^{-2} = \sum_k \beta_k \sigma_k^{-2}
$$

paper config:**top-M phi**(M=3 或 5,具体由 §6 验证决定),no baseline。

---

## 10. CMA-ES Inverse(修订:bounds 是 top-M cluster box 并集)

$$
\hat{\boldsymbol{\theta}} = \arg\min_{\boldsymbol{\theta} \in \bigcup_{k \in \text{top-M}}\mathcal{B}_k} \frac{1}{K_s}\sum_{\kappa=1}^{K_s} \frac{\|\hat{\mathbf{y}}_{\text{rBCM}}(\boldsymbol{\theta}; W_\kappa, H_\kappa) - \mathbf{y}_\kappa\|_2^2}{\|\mathbf{y}_\kappa\|_2^2} + \lambda R(\boldsymbol{\theta})
$$

**搜索 box 是 top-M cluster 的并集**——不再被 hard top-1 锁死。

---

## 11. Paper-ready 段落(修订版)

### 11.1 Method §"Hierarchical Ridge-Aware Gating"

> Our two-level Bayesian Gaussian Mixture (BGM) gate factorises the routing probability as $p(g, k \mid W, H, \mathbf{y}) = p(g \mid W, H)\,p(k \mid \boldsymbol{\phi}(\mathbf{y}, W, H), g)$, reflecting the causal structure that container geometry $(W, H)$ is an exogenous experimental variable while flow response $\mathbf{y}$ depends on the unknown material.
>
> The level-2 partition extends the standard $\boldsymbol{\phi}$-space BGM with a ridge-aware augmentation derived from Hamamichi et al. [2023]. Their plane-Poiseuille mapping $\boldsymbol{\phi}^{PP}: (\boldsymbol{\theta}, W, H) \mapsto (P, L)$ approximates an invariant under similarity-equivalent material substitutions (Hamamichi §5.4 footnote 15: the approximation is direction-faithful but not condition-number-faithful). At training time we form the joint feature $(\boldsymbol{\phi}_i, \boldsymbol{\phi}^{PP}_i)$ and fit a per-geo BGM with block-diagonal per-cluster covariance:
> $$p_g(\boldsymbol{\phi}, \boldsymbol{\phi}^{PP}) = \sum_k \pi_k^{(g)}\, \mathcal{N}(\boldsymbol{\phi}; \boldsymbol{\mu}_k^\phi, \boldsymbol{\Sigma}_k^\phi)\,\mathcal{N}(\boldsymbol{\phi}^{PP}; \boldsymbol{\mu}_k^{PP}, \boldsymbol{\Sigma}_k^{PP})$$
> Cluster membership $C_k^{(g)}$ thus collapses along similarity ridges in the proxy sense: each cluster's training data approximately spans a single ridge equivalence class instead of an arbitrary LHS-determined sub-segment. At inference, where $\boldsymbol{\theta}$ is unknown, we marginalise over $\boldsymbol{\phi}^{PP}$. In contrast to v1's claim, we do not assume the marginal $p(k\mid\phi)$ to be unimodal — when it is multi-modal (a verified failure mode in our data), we route to the top-M clusters by marginal posterior and aggregate predictions via posterior-weighted rBCM.

### 11.2 Method §"Why Ridge-Awareness (Approximately) Reduces Out-of-Box Failures"

> Without ridge-awareness, the level-2 BGM partitions purely by $\boldsymbol{\phi}$ similarity. Two materials on the same Hamamichi ridge produce nearly identical $\boldsymbol{\phi}$ and are routed to the same cluster — but their $\boldsymbol{\theta}$ values can lie at opposite ends of the ridge. With LHS-uniform training sampling, a typical cluster ends up containing only a sub-segment of its associated ridge, so the cluster's training-data bounding box covers only that segment.
>
> Ridge-aware joint training pulls all training samples on the same ridge (in the φ^PP proxy sense) into the same cluster, expanding the box to span the full equivalence class **conditional on training-data presence on the ridge**. This mitigates but does not eliminate the out-of-box failure: residual failures occur when (a) the φ^PP approximation is locally poor (near n→1, σ_y→0), or (b) training data has no support on a real material's ridge.

### 11.3 Method §"Inference-time Aggregation"

> We extend rBCM aggregation to incorporate BGM posterior uncertainty: the per-expert weight is $\beta_k = p(k\mid\phi, g)\cdot \tfrac{1}{2}[\log\sigma_{\text{prior}}^2 - \log\sigma_k^2]_+$, where the marginal posterior $p(k\mid\phi, g)$ — previously discarded by v2's variance-only weighting — propagates routing uncertainty into the predictive aggregate. This change is necessary even before ridge-aware training: it allows the inverse solver to escape mistakenly-confident routings via top-M aggregation, with confidence-weighted contribution from each candidate.

### 11.4 Method §"Honest Limitations"

> Hamamichi similarity is fundamentally information-theoretic at the single-setup limit: regardless of routing strategy, one cannot recover the ridge-tangent component of $\boldsymbol{\theta}^*$ from a single $(W, H, \mathbf{y})$ tuple. Our ridge-aware gate transforms the routing failure mode from "wrong cluster (no recovery possible)" to "correct ridge family (recovery up to ridge ambiguity, resolvable by 2-setup observations)." Failures persist when (i) the proxy $\phi^{PP}$ degenerates locally, (ii) training data lacks coverage on the true material's ridge, or (iii) the posterior $p(k\mid\phi)$ is sufficiently multi-modal that even top-M aggregation does not include the truth-bearing cluster.

---

## 12. 总结

**v1 → v2 主要修订**:
- **数学声称弱化**:Theorem 1 → Approximation Lemma A1;Theorem 2 → conditional Theorem 2'
- **新增隐患**:边缘化 multimodality(§3),rBCM bug(§4)
- **实施路径换序**:从"立刻重训"改为"先修 rBCM,再 dry-run,验证后才重训"(§5)
- **评价指标升级**:top-1 accuracy → top-M coverage + 物理 RMSE(§6)
- **数据集升级**:v3 → v4 = v3 + 4,177 行 hier-BO infill(§7)
- **下游模块同步修订**:rBCM 加 BGM posterior weight(§9),CMA bounds 改 top-M 并集(§10)

**Paper 立场**:不再宣称"我们彻底解决路由错误",改为"我们把路由错误的失败模式从'数据不可救'转为'2-setup 可救',并诚实标注 proxy 局限"。这是更弱但更可辩护的声明。

---

## 附录 A:对原 v1 spec 的逐项修订映射

| v1 段落 | v2 修订 | 原因(对应质疑) |
|---|---|---|
| §2.3 Theorem 1 | Lemma A1(approximation) | φ^PP 是 PP-近似 proxy,Hamamichi 自承不严格 |
| §2.3 "BIC 自动平衡 不需要 λ" | "BIC 不直接对应 inverse 精度" | BIC 是密度拟合,非真值恢复 |
| §2.3 "推理代码零改动" | "形式同 v2 但需 dry-run 验证 marginal multimodality" | 边缘化后 cluster 仍可能在 φ 重叠 |
| §2.4 Theorem 2 + Corollary | Theorem 2'(conditional) | 训练数据沿 ridge 不连续时 box 不一定覆盖 |
| §4 rBCM 公式 | 加 $p(k\mid\phi)$ weight | rBCM 漏 BGM posterior 是 v2 既存 bug |
| §5 CMA bounds | top-M 并集 | hard top-1 不合 single-setup 不可辨识 |
| §6 paper 段落 | 全部弱化 | 配合上述 |
| 实施路径(立刻 Phase 4 重训) | A→B→C→D→E→F 验证-优先 | 先证明 ridge-aware 真有用再投 9h |
