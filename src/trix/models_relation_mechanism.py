import torch
from torch import nn

from .models_relation import RelNet, EntityNet


class TRIXLatentMechanism(nn.Module):
    """
    Query-free 版本的 TRIX，显式建模潜在机制 z：
    - 不使用真实关系初始化 query，避免 train/test 条件不一致和信息泄漏
    - 用 (h,t) 的上下文节点表征编码高斯隐变量 z
    - 关系候选特征来自关系图更新，打分时拼接 z
    - forward 返回 logits，并把 KL 缓存在 self.last_kl（训练 loop 可直接读取）
    """

    def __init__(self, rel_model_cfg, entity_model_cfg, trix_cfg, mechanism_cfg=None):
        super().__init__()
        mechanism_cfg = mechanism_cfg or {}

        self.num_layer = trix_cfg.num_layer
        self.feature_dim = trix_cfg.feature_dim
        self.num_mlp_layer = trix_cfg.num_mlp_layer
        self.z_dim = mechanism_cfg.get("z_dim", self.feature_dim)
        self.deterministic_eval = mechanism_cfg.get("deterministic_eval", True)

        # 关系/实体更新网络（复用原 TRIX 的 RelNet / EntityNet）
        self.relation_model = nn.ModuleList()
        self.entity_model = nn.ModuleList()
        for _ in range(self.num_layer):
            self.relation_model.append(RelNet(**rel_model_cfg))
            self.entity_model.append(EntityNet(**entity_model_cfg))

        # query token（不依赖真实关系），relation token（初始关系表示）
        self.query_token = nn.Parameter(torch.zeros(self.feature_dim))
        self.relation_token = nn.Parameter(torch.zeros(self.feature_dim))

        # 机制编码器：f(ht) -> (mu, logvar)
        encoder_in_dim = self.feature_dim * 4  # h, t, h-t, h*t
        self.mech_mu = nn.Sequential(
            nn.Linear(encoder_in_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.z_dim),
        )
        self.mech_logvar = nn.Sequential(
            nn.Linear(encoder_in_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.z_dim),
        )

        # 打分 MLP：concat(rel_feat, z) -> logit
        score_in_dim = self.feature_dim + self.z_dim
        self.score_mlp = nn.Sequential(
            nn.Linear(score_in_dim, score_in_dim),
            nn.ReLU(),
            nn.Linear(score_in_dim, 1),
        )

        self.last_kl = None

    def forward(self, data, batch):
        """
        Args:
            data: 图数据，包含 relation_adj 等
            batch: [B, R, 3]，其中 R=num_relations//2（所有候选关系），列 0 是正例
        Returns:
            logits: [B, R]，与现有训练/评估接口兼容
        """
        h_index, t_index, r_index = batch.unbind(-1)  # [B, R]
        h_index = h_index.long()
        t_index = t_index.long()
        r_index = r_index.long()
        # 保留候选关系 id（用于负采样和最终 gather），但下方初始化 query 使用独立的 token，不使用 r_index
        batch = torch.stack([h_index, t_index, r_index], dim=-1)
        rel_graph = data.relation_adj
        device = h_index.device
        batch_size = h_index.shape[0]
        num_relations = rel_graph["hh"].num_nodes

        # 初始化关系表示：query-free，使用共享 relation_token
        relation_representations = self.relation_token.view(1, 1, -1).expand(batch_size, num_relations, -1)

        # 初始化节点表示：query token 只依赖 (h,t)，不依赖真实关系
        node_representations = torch.zeros(batch_size, data.num_nodes, self.feature_dim, device=device)
        query = self.query_token.view(1, 1, -1).expand(batch_size, 1, -1)
        h_pos = h_index[:, 0].long().view(batch_size, 1, 1)
        t_pos = t_index[:, 0].long().view(batch_size, 1, 1)
        node_representations.scatter_add_(1, h_pos.expand(-1, 1, self.feature_dim), query)
        node_representations.scatter_add_(1, t_pos.expand(-1, 1, self.feature_dim), -query)

        # 交替更新实体/关系表示（复用原 TRIX 的层次）
        for i in range(self.num_layer):
            node_representations = self.entity_model[i](data, batch, node_representations, relation_representations)
            relation_representations = self.relation_model[i](data, batch, relation_representations, node_representations)

        # 机制编码：用最终的节点表示（正例位置）
        h_vec = node_representations[torch.arange(batch_size, device=device), h_index[:, 0]]
        t_vec = node_representations[torch.arange(batch_size, device=device), t_index[:, 0]]
        f_ht = torch.cat([h_vec, t_vec, h_vec - t_vec, h_vec * t_vec], dim=-1)

        mu = self.mech_mu(f_ht)
        logvar = self.mech_logvar(f_ht)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        if self.training or not self.deterministic_eval:
            z = mu + eps * std
        else:
            z = mu

        # KL（按 batch 均值），供训练 loop 读取
        self.last_kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)).mean()

        # 关系候选特征按候选 id 对齐
        rel_feat = relation_representations.gather(1, r_index.unsqueeze(-1).expand(-1, -1, relation_representations.shape[-1]))
        z_expand = z.unsqueeze(1).expand(-1, rel_feat.shape[1], -1)
        logits = self.score_mlp(torch.cat([rel_feat, z_expand], dim=-1)).squeeze(-1)
        return logits

