import torch
import torch.nn as nn
import torch.nn.functional as F

class PairwiseEmbedding(nn.Module):
    """
    Computes pairwise features (U_ij) from particle features.
    Used to augment the attention weights with physical distance info.
    """
    def __init__(self, input_dim, out_dim):
        super().__init__()
        # We process (xi, xj, xi-xj)
        self.net = nn.Sequential(
            nn.Linear(input_dim * 3, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        # x: (B, N, C)
        B, N, C = x.shape
        
        # 1. Create pair combinations
        # x_i: (B, N, 1, C) -> repeats along dim 2
        x_i = x.unsqueeze(2).expand(B, N, N, C)
        # x_j: (B, 1, N, C) -> repeats along dim 1
        x_j = x.unsqueeze(1).expand(B, N, N, C)
        
        # 2. Build pairwise feature vector: (x_i, x_j, x_i - x_j)
        # Shape: (B, N, N, 3*C)
        pair_feat = torch.cat([x_i, x_j, x_i - x_j], dim=-1)
        
        # 3. Project to embedding dimension
        return self.net(pair_feat) # (B, N, N, out_dim)

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
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # MLP block
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Project pairwise interaction to attention heads
        self.pair_proj = nn.Linear(embed_dim, num_heads)

    def forward(self, x, u_ij=None):
        # x: (B, N, C)
        # u_ij: (B, N, N, C) - Pairwise features
        
        B, N, C = x.shape
        
        # --- 1. Attention ---
        residual = x
        x = self.norm1(x)
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # (B, Heads, N, Head_Dim)
        
        # Dot product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale # (B, Heads, N, N)
        
        # ADD PAIRWISE BIAS 
        if u_ij is not None:
            # Project u_ij to (B, N, N, Heads) then permute to (B, Heads, N, N)
            pair_bias = self.pair_proj(u_ij).permute(0, 3, 1, 2)
            attn = attn + pair_bias
            
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
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
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # MLP block
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: (B, N+1, C) where index 0 is CLS token
        
        B, N_plus_1, C = x.shape
        N = N_plus_1 - 1
        
        # --- 1. Attention ---
        residual = x
        x = self.norm1(x)
        
        qkv = self.qkv(x).reshape(B, N_plus_1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # (B, Heads, N+1, Head_Dim)
        
        # Only CLS token (index 0) attends to others
        q_cls = q[:, :, 0:1, :] # (B, Heads, 1, Head_Dim)
        attn = (q_cls @ k.transpose(-2, -1)) * self.scale # (B, Heads, 1, N+1)
        
        attn = attn.softmax(dim=-1)
        x_cls = (attn @ v).transpose(1, 2).reshape(B, 1, C) # (B, 1, C)
        x_cls = self.proj(x_cls)
        
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
    def __init__(self, input_dim, embed_dim=128, num_heads=8, num_layers=8, num_cls_layers=2, dropout=0.1, num_classes=1):
        super().__init__()
        
        # Input Embedding
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.embed = nn.Linear(input_dim, embed_dim)
        self.pair_embed = PairwiseEmbedding(input_dim, embed_dim)
        
        # Class Token (learnable vector)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer Blocks
        self.part_atten_blocks = nn.ModuleList([
            ParticleAttentionBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.cls_atten_blocks = nn.ModuleList([
            ClassAttentionBlock(embed_dim, num_heads, dropout)
            for _ in range(num_cls_layers)
        ])
        
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
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x: (B, N, C)
        B = x.shape[0]
        
        # 1. Preprocessing
        # BN expects (B, C, N), so transpose
        x = x.transpose(1, 2)
        x = self.input_bn(x)
        x = x.transpose(1, 2) # Back to (B, N, C)
        
        # 2. Compute Pairwise Features (before adding CLS token)
        # This is expensive (N^2), so we do it once
        u_ij = self.pair_embed(x) # (B, N, N, Embed)
        
        # Pad u_ij for the CLS token
        # The CLS token (index 0) has no geometric relation to others, so bias is 0
        # Shape becomes (B, N+1, N+1, Embed)
        u_ij_padded = F.pad(u_ij, (0, 0, 1, 0, 1, 0), value=0.0)
        
        # 3. Embedding & CLS Token
        x = self.embed(x) # (B, N, Embed)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # (B, N+1, Embed)
        
        # 4. Transformer Layers
        for block in self.part_atten_blocks:
            x = block(x, u_ij_padded)
            
        for block in self.cls_atten_blocks:
            x = block(x)
            
        x = self.norm(x)
        
        # 5. Extract CLS token for classification
        cls_output = x[:, 0]
        return self.head(cls_output)