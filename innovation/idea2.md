# idea2：Relations as Latent Generative Mechanisms（面向 TRIX 代码可直接实现的实验方案）

> 目标：把 `innovation/rrecord.md` 里“关系不是标签，而是潜在生成机制（latent mechanism）”的想法，落成 **TRIX 当前关系预测管线里可跑的模型与实验**。本文件按“要改哪里 / 怎么实现 / 怎么跑对照与消融”写，交给 code agent 直接开工即可。

---

## -1. 理论解释（建议写进论文的 Theory / Motivation）

这一节回答三个问题：

1) **为什么现有关系预测（把关系当 label）在理论上必然失败**  
2) **为什么“关系=潜在生成机制 z”在表达力上严格更强**  
3) **什么才是“正确推荐”的实现方式**（避免信息泄漏、保证训练/测试一致、让机制而不是 ID 被学到）

### -1.1 任务形式化：关系预测其实是“对 (h,t) 的机制反演”

给定图 \(G=(V,R,E)\)，以及一对实体 \((h,t)\)，关系预测任务是从候选集合里选关系：

\[
\hat r = \arg\max_{r\in\mathcal{R}} s(h,t,r;G)
\]

现有方法（包含大量 inductive KGE / GNN / ULTRA / TRIX 风格）可以抽象成：

\[
s_{\text{label}}(h,t,r;G)= g_\theta\big(\Phi(G,h,t),\, \psi(r)\big)
\]

- \(\Phi(G,h,t)\)：模型从图里“可见”的结构信息（局部邻域、路径统计、关系共现、计数等），通常受限于 message passing / WL 范式
- \(\psi(r)\)：把关系当作“候选类别/节点”的表示（哪怕 fully-inductive 不用显式 ID，也依赖“关系作为离散对象”的建模接口）

**机制视角**重构为：关系不是输入类别，而是驱动 \(h\rightarrow t\) 连接的潜在过程 \(z\)。

\[
z_{h,t}\sim p_\theta(z\mid G,h,t),\quad
s_{\text{mech}}(h,t,r;G)= \mathbb{E}_{z_{h,t}}\Big[g_\theta\big(\Phi(G,h,t,r), z_{h,t}\big)\Big]
\]

这句话是核心：**“关系预测 = 反演生成机制”**，不是“分类一个标签”。

### -1.2 不可区分性：结构等价 \(\Rightarrow\) label 模型不可避免地给出相同输出

定义一个模型能看到的“结构视野” \(\mathcal{N}_k(h,t)\)（例如 k-hop 子图、路径计数、关系共现直方图等）。如果两个样本 \((G_1,h_1,t_1)\) 与 \((G_2,h_2,t_2)\) 在该视野下同构（对模型而言等价），则对任意仅依赖该视野的模型都有：

\[
\Phi(G_1,h_1,t_1)=\Phi(G_2,h_2,t_2)\ \Rightarrow\ 
s_{\text{label}}(h_1,t_1,r;G_1)=s_{\text{label}}(h_2,t_2,r;G_2)\quad \forall r
\]

因此，一旦真实世界存在“**结构等价但语义/机制不同**”的关系（例如 causally_affects vs correlated_with），任何只靠结构统计的 label 化模型都会发生**不可约失败**：不是容量不够，而是信息不足。

这就是你 `rrecord.md` 里“结构 = 多种机制叠加投影”的理论根源。

### -1.3 机制建模为何严格更强：从“点表示”提升到“分布表示”

label 模型本质上把每个 \((h,t)\) 映射到一个确定的表示（或确定的 logits）：

\[
f_{\text{label}}(h,t)=g(\Phi(G,h,t))
\]

机制模型把 \((h,t)\) 映射到一个**机制分布** \(p(z\mid G,h,t)\)，再通过积分/期望做预测：

\[
f_{\text{mech}}(h,t)=\int g(z,\Phi(G,h,t,\cdot))\ p(z\mid G,h,t)\, dz
\]

直观上：

- label：每个样本一个“点”
- mechanism：每个样本一个“分布/混合”

分布能表达：

- **多机制叠加**（同一结构可由不同机制生成）
- **不确定性**（结构信息不足时，输出应不确定）
- **可迁移的因素分解**（跨域迁移更容易对齐到 z，而非对齐到关系名/ID）

因此可以提出一个常见的表达力命题（写成 theorem sketch 即可）：

\[
\mathcal{F}_{\text{label}} \subsetneq \mathcal{F}_{\text{mech}}
\]

### -1.4 identifiability 与“正确实现”的关键：训练/测试必须一致，且不能把真关系喂进 encoder

你当前代码里有一个必须正视的事实（在 `src/trix/models_relation.py`）：

- 训练时 `negative_sampling_relation` 会把真关系放到候选第 0 位，因此 `r_index[:,0]=r_\text{true}`
- 测试时 `all_negative_relation` 的第 0 位通常是关系 0，因此 `r_index[:,0]=0`

如果模型在 forward 里使用 `r_index[:,0]` 初始化 query，那么：

1) **训练/测试条件不一致**（train conditioned on truth, test conditioned on 0）  
2) **潜在信息泄漏**：训练时 encoder/表示可能“偷看”真关系，任何增益都可能是“泄漏收益”，理论叙事会直接崩。

因此：**正确推荐方案必须满足 query-free**：

- 机制 \(z_{h,t}\) 的 encoder 只能用 \((G,h,t)\)（以及不依赖真关系的上下文），不能用 \(r_\text{true}\)
- train/test 必须使用相同的 query 初始化逻辑

这不是“工程细节”，而是你的理论主张成立的前提。

---

## 0. 先对齐你仓库里“关系预测”的现状（非常关键）

### 0.1 训练/评估入口

- 入口：`src/run_relation.py`
- 训练：
  - batch 是三元组 `(h,t,r)` 组成的张量（shape: `[B, 3]`）
  - `tasks.negative_sampling_relation(...)` 会把每个样本扩展成 **在“所有关系”上的候选集合**（shape: `[B, R, 3]`，其中 `R = num_relations//2`），并把正例关系交换到 index=0
  - 模型 forward 输出 logits（shape: `[B, R]`），loss 用 `BCEWithLogits`，target 第 0 列为 1，其余为 0
- 测试：
  - `tasks.all_negative_relation` 为每个 `(h,t)` 枚举所有候选关系，模型输出 `[B, R]`
  - 指标用 `tasks.compute_ranking_relation`

### 0.2 重要细节：模型 forward 在训练/测试的条件可能不一致

当前 `src/trix/models_relation.py` 的 `TRIX.forward` 会用 `r_index[:,0]`（候选集合第一个关系）来初始化 query，并且 `RelNet.forward` 也会用 `batch[:,0,2]` 当作“query id”。

- 训练时：`negative_sampling_relation` 把**真关系**放在 index=0，所以 `r_index[:,0] = r_true`（有“信息泄漏”的风险）
- 测试时：`all_negative_relation` 生成的第一个关系通常是 `0`，所以 `r_index[:,0] = 0`（与训练不一致）

这对“机制建模”的公平性和稳定性影响很大：我们希望 latent mechanism **只由 (h,t,graph context) 决定**，而不是由真关系决定。

因此，本 idea 的实现会明确给出两条路线：

- **路线 A（最小改动、最快出结果）**：在现有 TRIX 表示上加 latent z，但**保持 TRIX 本体不变**，用于先验证“z 是否有增益”
- **路线 B（推荐、理论更一致）**：把 TRIX 的 query 改成与 relation label 无关（常量/learnable token），让表示真正只依赖 (h,t)，再叠加 latent z

---

## 1. 核心建模：关系 = 潜在生成机制 z

### 1.1 任务形式（与现有评估完全兼容）

对每个样本只给 `(h,t)`，在候选关系集合 \( \mathcal{R} \) 上打分：

\[
s_r(h,t) = \text{score}(z_{h,t}, \phi_r(h,t))
\]

- \(z_{h,t}\)：潜在机制（latent mechanism），**每个 (h,t) 一个**，不依赖关系标签
- \(\phi_r(h,t)\)：TRIX 现有网络产生的“候选关系特征”（当前代码里就是 `relation_representations` gather 出来的向量）
- `score`：MLP / 双线性 / 点积等

输出 logits 仍是 `[B, R]`，所以训练/测试代码可以最大程度复用。

---

## 1.2 正确推荐的方案（Strongly Recommended）：Query-free Mechanism TRIX（QFM-TRIX）

这一节给出我认为最“正确”的推荐：它既符合上面的理论叙事，也能在你仓库现有评估协议下直接跑。

### 1.2.1 核心原则

- **P1（query-free）**：任何用于构造节点/关系表示、以及 \(z_{h,t}\) 的 encoder，不能依赖 `r_index[:,0]` 或任何真关系注入
- **P2（任务兼容）**：输出仍是 `[B, R]` 的关系 logits，直接复用 `negative_sampling_relation/all_negative_relation/compute_ranking_relation`
- **P3（机制分布，而非机制向量）**：默认用随机隐变量（高斯），否则很容易退化成“多一层 MLP 的容量提升”，审稿会质疑贡献

### 1.2.2 结构（推荐实现）

**(A) Query-free 初始化**

在新模型（不要改 baseline TRIX）中引入一个 learnable token：

- `query_token ∈ R^d`，对每个 batch 复制成 `[B,d]`
- 用该 token 初始化对 \(h,t\) 的注入（scatter 到 head/tail 位置）

这样 train/test 绝对一致，并消除真关系泄漏。

**(B) 机制编码器 q(z | G,h,t)**

从“与真关系无关”的 node 表示里取 `h_vec/t_vec`，用：

\[
f_{ht}=[h\ \|\ t\ \|\ (h-t)\ \|\ (h\odot t)]
\Rightarrow (\mu,\log\sigma^2)=\text{MLP}(f_{ht})
\]

**(C) 关系候选特征 \(\phi_r\)**

保持 TRIX 的 relation graph 更新逻辑，得到每个候选关系的特征向量 `rel_feat ∈ R^{B×R×d}`。

关键：这部分也必须与真关系无关（即 query-free）。

**(D) 机制条件打分**

用轻量的 concat-MLP 做：

\[
s_r=\text{MLP}([\,\phi_r\ \|\ z\,])
\]

### 1.2.3 目标函数（推荐：ELBO 风格）

训练仍用你现在的 BCE（正例在 index=0），但加 KL：

\[
\mathcal{L}=\mathcal{L}_{\text{bce}}+\beta\cdot \text{KL}\big(q_\phi(z\mid G,h,t)\ \|\ \mathcal{N}(0,I)\big)
\]

推荐超参：

- `z_dim = d = feature_dim`
- `beta = 1e-4 ~ 1e-3`（先小后大）
- eval 默认 `z=mu`（保证稳定、复现容易）

> 备注：你也可以加入“free-bits”或 KL warmup，但建议先跑通上面这一版作为强基线。

### 1.2.4 这套方案为什么“正确”

- **理论一致**：encoder 不看真关系，train/test 条件一致，符合“机制反演”的假设
- **强对照**：baseline TRIX 完全不动；QFM-TRIX 的增益更可信
- **易写论文**：贡献点可以清晰写为
  - （i）指出现有关系预测存在条件不一致/泄漏风险（至少对你的代码是事实）
  - （ii）提出 query-free + latent mechanism 的统一框架
  - （iii）在 inductive 关系预测上验证机制可迁移与失败模式修复

### 1.2.5 不推荐/错误方案（务必避免）

- **错误 1：继续使用 `r_index[:,0]` 初始化 query，然后再加 z**  
  这会导致 train 时 conditioning on 真关系、test 时 conditioning on 0（或某个固定关系），属于“协议不一致 + 潜在信息泄漏”。这种实现即使涨点，也很难支撑“机制更强”的理论主张。

- **错误 2：z 只是 deterministic 向量、且没有任何机制约束（KL/先验/正则）**  
  很容易退化成“增加参数容量的特征融合”，审稿倾向把它当作常规工程技巧，而不是范式变化。

- **错误 3：直接改 baseline TRIX 的 forward**  
  会让对照实验不干净（你无法说清楚增益来自哪里）。正确做法是：baseline TRIX 保持不动，新模型单独实现。

---

## 2. 可实现方案（优先推荐 VAE 风格：z 是高斯隐变量）

### 2.1 Mechanism Encoder：q(z | h,t,context)

我们复用 TRIX forward 过程中已经算出来的节点表示 `node_representations`（shape: `[B, num_nodes, d]`）。

从中取出：

- `h_vec = node_representations[batch_idx, h]`（shape: `[B, d]`）
- `t_vec = node_representations[batch_idx, t]`（shape: `[B, d]`）

构造 pair 特征（经验上很稳）：

\[
f_{ht} = [h\_vec \;\|\; t\_vec \;\|\; (h\_vec - t\_vec) \;\|\; (h\_vec \odot t\_vec)]
\]

用一个 MLP 预测高斯分布参数：

- `mu = MLP_mu(f_ht)`（shape: `[B, z_dim]`）
- `logvar = MLP_logvar(f_ht)`（shape: `[B, z_dim]`）
- 采样：`z = mu + exp(0.5*logvar) * eps`

推理（eval）时可选：

- **deterministic**：`z = mu`
- **stochastic**：采样 1 次或多次取平均 logits

### 2.2 Mechanism Decoder：p(r | z, TRIX-feature)

TRIX 已经为每个候选关系给出特征向量：

- `rel_feat = relation_representations.gather(1, r_index)`（当前代码已有，shape: `[B, R, d]`）

最直接的打分方式（强烈推荐先做这个）：

- `score = MLP_score(concat(rel_feat, z_expanded))`
  - `z_expanded = z.unsqueeze(1).expand(-1, R, -1)`
  - `concat` 后 shape: `[B, R, d + z_dim]`
  - 输出 `[B, R]`

备选（更轻量）：

- 双线性：`score = (rel_feat @ W) · z`（需要小心维度）
- 点积：`score = rel_proj(rel_feat) · z`

### 2.3 Loss：BCE + β·KL（与现有训练 loop 兼容）

保留现有 BCE（正例在 index=0）：

\[
\mathcal{L}_{\text{bce}} = \text{BCEWithLogits}(s, y)
\]

加 KL 正则（prior 取标准正态）：

\[
\mathcal{L}_{\text{kl}} = \frac{1}{B}\sum_i \text{KL}\big(\mathcal{N}(\mu_i, \sigma_i^2)\;||\;\mathcal{N}(0,I)\big)
\]

总 loss：

\[
\mathcal{L} = \mathcal{L}_{\text{bce}} + \beta \cdot \mathcal{L}_{\text{kl}}
\]

建议的超参起点：

- `z_dim = 32`（与 TRIX feature_dim 对齐）
- `beta = 1e-3` 或 `1e-4`（先小一点，避免 KL 把模型压扁）
- `num_samples = 1`

---

## 3. 与 TRIX 代码对接：改哪些文件（建议落点）

### 3.1 新增一个模型类：TRIXLatentMechanism（不破坏现有 TRIX baseline）

建议新增文件：

- `src/trix/models_relation_mechanism.py`

里面实现一个新类（示意）：

```python
class TRIXLatentMechanism(nn.Module):
    def __init__(..., mech_cfg):
        # 1) 复用 TRIX 的 relation_model / entity_model 堆叠（可以 copy TRIX.forward 的前半段）
        # 2) 新增:
        #    - mech_encoder_mlp -> (mu, logvar)
        #    - mech_score_mlp   -> logits
        # 3) forward 返回 logits，同时把 kl 存在 self.last_kl 里
```

为什么用 `self.last_kl`：

- `run_relation.py` 当前只期待 `pred = model(data, batch)` 返回 logits
- 最小侵入的改法：让 forward 仍返回 logits，但训练时额外读 `model.last_kl` 加进 loss

### 3.2 修改训练 loop：在 `run_relation.py` 里加 KL（两行级别）

在计算 `loss = BCE(...)` 之后加：

- `kl = getattr(parallel_model.module if ddp else parallel_model, "last_kl", 0.0)`
- `loss = loss + cfg.model.mechanism.beta * kl`

并且日志里打印 `bce/kl/total`。

### 3.3 让 run_relation 支持从 config 选择模型类（推荐）

现在 `run_relation.py` 写死了：

- `from trix.models_relation import TRIX`
- `model = TRIX(...)`

建议改成：

- `model_cls_name = cfg.model.class`（例如 `"TRIX"` / `"TRIXLatentMechanism"`）
- `getattr(trix.models_relation, ...)` 或 `import trix.models_relation_mechanism`

这样你只需要新增一个 yaml 就能切换模型。

---

## 4. 路线 B（推荐）：修正“query 依赖真关系”的问题，让机制建模更干净

如果你想让理论叙事更硬、实验更干净，建议让 forward 的“query token”不再依赖 `r_index[:,0]`。

### 4.1 做法 1：固定 query relation id = 0（最省事，但有点丑）

把：

- `query = relation_representations[range(B), r_index[:,0]]`

改成：

- `query = relation_representations[range(B), 0]`（恒为第 0 个 relation node）

同时 `RelNet.forward` 里不要用 `batch[:,0,2]` 作为 query id。

### 4.2 做法 2：引入 learnable “CLS relation token”（更合理）

在模型里加一个参数：

- `self.query_token = nn.Parameter(torch.zeros(d))`

然后用它初始化 node state（scatter 到 h/t）：

- `query = self.query_token.expand(B, d)`

这样 train/test 完全一致，且不携带 relation label 信息。

> 建议：路线 B 不要直接改 baseline TRIX，避免影响对照；而是只在新模型（mechanism版）里使用。

---

## 5. 实验设计：怎么证明“机制建模”有意义（可投稿风格）

### 5.1 主实验（必须）

- 数据集：优先跑 **inductive**（更契合“机制可迁移”叙事）
  - `FB15k237Inductive v1-v4`
  - `WN18RRInductive v1-v4`
  - `NELLInductive v1-v4`
  - `ILPC2022 small/large`
  - `Ingram` 系列
- 对照：
  - **TRIX baseline**（原样）
  - **TRIX + latent z（路线 A）**
  - **TRIX(query-free) + latent z（路线 B）**
- 指标：沿用当前代码 `mr/mrr/hits@k`

### 5.2 消融（建议至少做 4 个）

- **β=0**：去掉 KL（相当于 deterministic 机制向量），看增益是否来自 stochastic/regularization
- **z_dim**：`[16, 32, 64]`
- **score 形式**：`concat-MLP` vs `bilinear`
- **z 的输入特征**：
  - 只用 `(h_vec, t_vec)`
  - 加 `(h_vec - t_vec, h_vec ⊙ t_vec)`（推荐）

### 5.3 机制解释性（可选但很加分）

做一个简单的分析脚本：

- 对同一 `(h,t)` 不同数据集/子域，比较 `mu` 的分布（余弦相似度、聚类）
- 看 z 是否按“语义机制”聚类（比如生物KG里 binding/interaction 等）

---

## 6. 配置建议（给 code agent 的可直接复制模板）

建议新增一个 config：

- `config/run_relation_inductive_mech.yaml`

核心新增字段（示意）：

```yaml
model:
  class: TRIXLatentMechanism
  trix:
    feature_dim: 32
    num_layer: 3
    num_mlp_layer: 2
  relation_model: { ...同原来... }
  entity_model:   { ...同原来... }
  mechanism:
    enabled: true
    z_dim: 32
    beta: 1.0e-3
    deterministic_eval: true
    score_type: concat_mlp   # concat_mlp | bilinear | dot
    query_free: true         # 是否使用 CLS query token（路线B）
```

---

## 7. 跑实验命令（与现有 command_rel.md 对齐）

你现在 `command_rel.md` 的命令都是：

- `python src/run_relation.py -c ./config/run_relation_inductive.yaml ...`

机制版只需要换 config：

- `-c ./config/run_relation_inductive_mech.yaml`

其余 dataset/version/ckpt/gpus/epochs/bpe 都保持一致，确保对照公平。

---

## 8. 常见坑（提前写给 code agent，省大量时间）

- **关系数的定义**：代码里用的是 `num_relations // 2`（因为会有 inverse relation）。机制模型必须保持一致，否则 batch 维度/评估会错。
- **train/test 一致性**：务必处理 `r_index[:,0]` 在训练是 `r_true`、测试是 `0` 的问题（见路线 B）。
- **KL collapse**：如果 `beta` 太大，`mu` 会被压到 0，模型退化成 baseline。先从 `1e-4~1e-3` 开始。
- **速度/显存**：关系预测这里本来就枚举所有关系（R 可能不小），`score_mlp` 要尽量轻量（2层MLP足够）。

---

## 9. 交付清单（code agent 完成后你应该看到什么）

- 新增：`src/trix/models_relation_mechanism.py`（或等价文件）
- 修改：`src/run_relation.py`（支持新模型 + loss 加 KL）
- 新增：`config/run_relation_inductive_mech.yaml`（和可选的 transductive 版本）
- 输出日志里应包含：`bce / kl / total`，并能在 `output_rel/...` 下正常保存 ckpt 与测试指标

---

## 10. 最小可行实现（MVP）优先级（建议 code agent 按这个顺序做）

1) 路线 A：TRIX 表示不动，只加 z + KL（最快验证）
2) 路线 B：query-free（CLS token） + z（更干净的结论）
3) 消融：β、z_dim、score_type
4)（可选）机制解释性分析脚本



