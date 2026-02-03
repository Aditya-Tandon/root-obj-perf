import torch
import torch.nn as nn
import torch.nn.functional as F


def to_rapidity(x, eps=1e-8):
    mass = x[..., 0]
    pt = x[..., 1]
    eta = x[..., 2]
    rapidity = torch.log(
        (
            torch.sqrt(mass**2 + pt**2 * torch.cosh(eta) ** 2)
            + pt * torch.sinh(eta)
            + eps
        )
        / (torch.sqrt(mass**2 + pt**2) + eps)
    )
    return rapidity


def to_energy(x, eps=1e-8):
    mass = x[..., 0]
    pt = x[..., 1]
    eta = x[..., 2]
    energy = torch.sqrt(mass**2 + pt**2 * torch.cosh(eta) ** 2 + eps)
    return energy


def to_pt(x, eps=1e-8):
    px = x[..., 1]
    py = x[..., 2]
    pt = torch.sqrt(px**2 + py**2 + eps)
    return pt


class PairwiseEmbedding(nn.Module):
    """
    Computes pairwise features (U_ij) from particle features.
    Used to augment the attention weights with physical distance info.
    """

    def __init__(self, input_dim, out_dim):
        super().__init__()
        # to process concatenated delta, k_t, z, m^2 from the ParT paper
        self.net = nn.Sequential(
            nn.Linear(input_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x, particle_mask=None):
        B, N, C = x.shape

        if particle_mask is None:
            particle_mask = x[..., 0] != 0  # (B, N)
        # Get physical quantities
        r = to_rapidity(x).unsqueeze(2)
        phi = x[..., 3].unsqueeze(2)
        pt = x[..., 1].unsqueeze(2)
        E = to_energy(x).unsqueeze(2)

        delta = torch.sqrt(
            (r - r.transpose(1, 2)) ** 2 + (phi - phi.transpose(1, 2)) ** 2 + 1e-8
        )

        k_t = torch.min(pt, pt.transpose(1, 2)) * delta

        z = torch.min(pt, pt.transpose(1, 2)) / (pt + pt.transpose(1, 2) + 1e-8)

        # p = x[..., 1:4]  # (B, N, 3)
        # p_dot_p = torch.matmul(p, p.transpose(1, 2))  # (B, N, N)
        # p_sq = torch.sum(p**2, dim=-1, keepdim=True)  # (B, N, 1)
        # norm_p_sum_sq = p_sq + p_sq.transpose(1, 2) + 2 * p_dot_p
        # m_2 = (E + E.transpose(1, 2)) ** 2 - norm_p_sum_sq  # (B, N, N)
        m_2 = (
            2
            * pt
            * pt.transpose(1, 2)
            * (torch.cosh(r - r.transpose(1, 2)) - torch.cos(phi - phi.transpose(1, 2)))
            + 1e-8
        )

        # take the log of all features to compress the range
        delta = torch.log(torch.clamp(delta, min=1e-8))
        m_2 = torch.log(torch.clamp(m_2, min=1e-8))
        k_t = torch.log(torch.clamp(k_t, min=1e-8))
        z = torch.log(torch.clamp(z, min=1e-8))

        dxy = x[..., 4].unsqueeze(2)  # (B, N, 1)
        z0 = x[..., 5].unsqueeze(2)  # (B, N, 1)
        q = x[..., 6].unsqueeze(2)  # (B, N, 1)

        d_dxy = dxy - dxy.transpose(1, 2)  # (B, N, N)
        d_z0 = z0 - z0.transpose(1, 2)  # (B, N, N)

        pt_jet = pt.sum(dim=1, keepdim=True)  # (B, 1, 1)
        q = q * pt / (pt_jet + 1e-8)  # charge weighted by pt
        q_ij = q * q.transpose(1, 2)  # (B, N, N)

        u = torch.stack([delta, k_t, z, m_2, d_dxy, d_z0, q_ij], dim=-1)

        if particle_mask is not None:
            pairwise_mask = particle_mask.unsqueeze(1) & particle_mask.unsqueeze(
                2
            )  # (B, N, N)
            u = u * pairwise_mask.unsqueeze(-1).float()

        # Project to embedding dimension
        return self.net(u)  # (B, N, N, out_dim)


class ParticleAttentionBlock(nn.Module):
    """
    Standard Transformer Block modified to accept Pairwise Bias (U_ij).
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (embed_dim // num_heads) ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # MLP block
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

        # Project pairwise interaction to attention heads
        self.pair_proj = nn.Linear(embed_dim, num_heads)

    def forward(self, x, u_ij=None, particle_mask=None):
        # x: (B, N, C)
        # u_ij: (B, N, N, C) - Pairwise features
        # particle_mask: (B, N) - Mask for valid particles

        B, N, C = x.shape

        # --- 1. Attention ---
        residual = x
        x = self.norm1(x)

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)  # (3, B, Heads, N, Head_Dim=C//Heads)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, Heads, N, Head_Dim)

        # Dot product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, Heads, N, N)
        if particle_mask is not None:
            attn_mask = particle_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
            attn = attn.masked_fill(~attn_mask, float("-inf"))

        # ADD PAIRWISE BIAS
        if u_ij is not None:
            # Project u_ij to (B, N, N, Heads) then permute to (B, Heads, N, N)
            pair_bias = self.pair_proj(u_ij).permute(0, 3, 1, 2)
            attn = attn + pair_bias

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.attn_drop(x)
        x = x + residual

        # --- 2. MLP ---
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual

        return x


class ClassAttentionBlock(nn.Module):
    """
    Transformer Block for Class Token attention.
    Only the class token attends to the particle tokens.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (embed_dim // num_heads) ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # MLP block
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, particle_mask=None):
        # x: (B, N+1, C) where index 0 is CLS token
        # particle_mask: (B, N) - Mask for valid particles

        B, N_plus_1, C = x.shape
        N = N_plus_1 - 1

        # --- 1. Attention ---
        residual = x
        x = self.norm1(x)

        qkv = (
            self.qkv(x)
            .reshape(B, N_plus_1, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)  # (3, B, Heads, N+1, Head_Dim=C//Heads)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, Heads, N+1, Head_Dim)

        # Only CLS token (index 0) attends to others
        q_cls = q[:, :, 0:1, :]  # (B, Heads, 1, Head_Dim)
        attn = (q_cls @ k.transpose(-2, -1)) * self.scale  # (B, Heads, 1, N+1)

        if particle_mask is not None:
            attn_mask = torch.cat(
                [
                    torch.ones(
                        (B, 1, 1, 1),
                        dtype=particle_mask.dtype,
                        device=particle_mask.device,
                    ),
                    particle_mask.unsqueeze(1).unsqueeze(2),
                ],
                dim=-1,
            )  # (B, 1, 1, N+1)
            attn = attn.masked_fill(~attn_mask, float("-inf"))

        attn = attn.softmax(dim=-1)
        x_cls = (attn @ v).transpose(1, 2).reshape(B, 1, C)  # (B, 1, C)
        x_cls = self.proj(x_cls)
        x_cls = self.attn_drop(x_cls)

        # Replace CLS token embedding
        x = torch.cat([x_cls, x[:, 1:, :]], dim=1)
        x = x + residual

        # --- 2. MLP ---
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual
        return x


class ParticleTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_dim=128,
        num_pairwise_feat=7,
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        dropout=0.1,
        num_classes=1,
        use_batch_norm=True,  # Set to False for small batch overfit tests
    ):
        super().__init__()

        # Input Embedding - use LayerNorm for small batches, BatchNorm otherwise
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.input_bn = nn.BatchNorm1d(input_dim)
        else:
            self.input_ln = nn.LayerNorm(input_dim)
        self.embed = nn.Linear(input_dim, embed_dim)
        self.pair_embed = PairwiseEmbedding(num_pairwise_feat, embed_dim)

        # Class Token (learnable vector)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Transformer Blocks
        self.part_atten_blocks = nn.ModuleList(
            [
                ParticleAttentionBlock(embed_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        self.cls_atten_blocks = nn.ModuleList(
            [
                ClassAttentionBlock(embed_dim, num_heads, dropout)
                for _ in range(num_cls_layers)
            ]
        )

        # Final Norm
        self.norm = nn.LayerNorm(embed_dim)

        # Classification Head
        layers = []
        for _ in range(num_cls_layers - 1):
            layers.append(nn.Linear(embed_dim, embed_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(embed_dim, num_classes))
        self.head = nn.Sequential(*layers)

        # Init weights
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, particle_mask=None):
        # x: (B, N, C)
        B = x.shape[0]

        # 1. Save raw input for pairwise features (needs physical interpretation)
        x_raw = x

        # 2. Preprocessing - use appropriate normalization
        if self.use_batch_norm:
            # BN expects (B, C, N), so transpose
            x = x.transpose(1, 2)
            x = self.input_bn(x)
            x = x.transpose(1, 2)  # Back to (B, N, C)
        else:
            x = self.input_ln(x)  # LayerNorm works on last dim directly

        # 3. Compute Pairwise Features (before adding CLS token)
        # This is expensive (N^2), so we do it once
        u_ij = self.pair_embed(x_raw, particle_mask)  # (B, N, N, Embed)

        # 4. Embedding and add CLS token
        x = self.embed(x)  # (B, N, Embed)
        for block in self.part_atten_blocks:
            x = block(x, u_ij, particle_mask)

        # 5. Add CLS token and run class attention blocks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, N+1, Embed)

        # Class attention: CLS token attends to all particle tokens
        for block in self.cls_atten_blocks:
            x = block(x, particle_mask)

        x = self.norm(x)

        # 6. Extract CLS token for classification
        cls_output = x[:, 0]
        return self.head(cls_output)
