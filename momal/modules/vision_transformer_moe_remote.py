import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


# ============================
# === Missing-aware Adapter ===
# ============================
class MaMOLAdapter(nn.Module):
    """
    Missing-aware Mixture-of-Loras Adapter: Δh = h_dyn + h_stat
    """
    def __init__(self, hidden_dim, rank=8, num_dyn=2, num_stat=3, topk=1, router_dim=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.num_dyn = num_dyn
        self.num_stat = num_stat
        self.topk = topk
        self.router_dim = router_dim

        if num_dyn > 0:
            self.A_dyn = nn.ParameterList([
                nn.Parameter(torch.randn(hidden_dim, rank) * 0.02) for _ in range(num_dyn)
            ])
            self.B_dyn = nn.ParameterList([
                nn.Parameter(torch.randn(rank, hidden_dim) * 0.02) for _ in range(num_dyn)
            ])
            self.router = nn.Sequential(
                nn.Linear(hidden_dim + router_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, num_dyn),
            )
        else:
            self.A_dyn = nn.ParameterList()
            self.B_dyn = nn.ParameterList()
            self.router = None
       
        self.num_stat = num_stat
        if num_stat > 0:
            self.A_stat = nn.ParameterList([
                nn.Parameter(torch.randn(hidden_dim, rank) * 0.02) for _ in range(num_stat)
            ])
            self.B_stat = nn.ParameterList([
                nn.Parameter(torch.randn(rank, hidden_dim) * 0.02) for _ in range(num_stat)
            ])
        else:
            self.A_stat = nn.ParameterList()
            self.B_stat = nn.ParameterList()

        self.act = nn.GELU()
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def lora(self, x, A, B):
        return (x @ A) @ B

    def forward(self, x, missing_embed):
        L, B, D = x.shape
        m = missing_embed.unsqueeze(0).expand(L, -1, -1)

        dyn = torch.zeros_like(x)
        if self.num_dyn > 0:
            gates = torch.softmax(self.router(torch.cat([x, m], dim=-1)), dim=-1)
            topv, topi = torch.topk(gates, k=min(self.topk, self.num_dyn), dim=-1)
            for i in range(self.topk):
                idx = topi[..., i]
                val = topv[..., i].unsqueeze(-1)
                for k in range(self.num_dyn):
                    mask = (idx == k).float().unsqueeze(-1)
                    delta = self.lora(x, self.A_dyn[k], self.B_dyn[k])
                    dyn += val * mask * delta

        shared = torch.zeros_like(x)
        spec = torch.zeros_like(x)

        if self.num_stat > 0:
            shared = self.lora(x, self.A_stat[0], self.B_stat[0])

            for j in range(1, self.num_stat):
                if j < missing_embed.shape[1]:
                    act = (missing_embed[:, j] > 0.5).float().view(1, B, 1)
                    spec += act * self.lora(x, self.A_stat[j], self.B_stat[j])


        delta_h = self.out_proj(self.act(dyn + shared + spec))
        return delta_h


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head, attn_mask=None, mamol_enable=True, mamol_cfg=None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.ln_2 = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", nn.GELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        for p in self.mlp.parameters():
            p.requires_grad = False

        self.mamol = None
        if mamol_enable:
            cfg = mamol_cfg or dict(rank=8, num_dyn=4, num_stat=3, topk=2, router_dim=8)
            self.mamol = MaMOLAdapter(
                hidden_dim=d_model,
                rank=cfg["rank"],
                num_dyn=cfg["num_dyn"],
                num_stat=cfg["num_stat"],
                topk=cfg["topk"],
                router_dim=cfg["router_dim"],
            )

        self.attn_mask = attn_mask

    def attention(self, x):
        mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=mask)[0]

    def forward(self, x, missing_embed):
        x = x + self.attention(self.ln_1(x))
        z = self.ln_2(x)
        h_frozen = self.mlp(z)
        delta = self.mamol(z, missing_embed) if self.mamol is not None else 0
        return x + h_frozen + delta


# ============================
# === Transformer Stack ======
# ============================
class Transformer(nn.Module):
    def __init__(self, width, layers, heads, attn_mask=None,
                 mamol_enable=True, mamol_cfg=None, mamol_layers=None):
        super().__init__()
        self.width = width
        self.layers = layers

        self.mamol_layers = set(mamol_layers or [])

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                width, heads, attn_mask,
                mamol_enable=(mamol_enable and (i in self.mamol_layers)),
                mamol_cfg=mamol_cfg
            ) for i in range(layers)
        ])

    def forward(self, x, missing_embed):
        for blk in self.resblocks:
            x = blk(x, missing_embed)
        return x



# ============================
# === Vision Transformer =====
# ============================
class VisionTransformer(nn.Module):
    def __init__(self, input_resolution, patch_size, width, layers, heads, output_dim,
                 mamol_enable=True, mamol_cfg=None, mamol_layers=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(3, width, patch_size, patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width)
        )
        self.ln_pre = nn.LayerNorm(width)

        # Router embedding
        router_dim = (mamol_cfg or {}).get("router_dim", 8)
        self.missing_proj = nn.Linear(3, router_dim)

        # Transformer with partial MaMOL injection
        self.transformer = Transformer(
            width, layers, heads,
            mamol_enable=mamol_enable,
            mamol_cfg=mamol_cfg,
            mamol_layers=mamol_layers,
        )

        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))


    def forward(self, x, missing_type=None):
        # x = self.conv1(x)
        # x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = self.class_embedding.to(x.dtype)
        cls = cls + torch.zeros(x.shape[0], 1, cls.shape[0], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        if isinstance(missing_type, list):
            missing_type = torch.tensor(missing_type, device=x.device)
        elif not isinstance(missing_type, torch.Tensor):
            missing_type = torch.as_tensor(missing_type, device=x.device)
        # === missing_type to embedding ===
        if missing_type is None:
            missing_type = torch.zeros(x.shape[1], dtype=torch.long, device=x.device)
        mt_oh = F.one_hot(missing_type.to(torch.int64), num_classes=3).float()
        missing_embed = self.missing_proj(mt_oh)

        x = self.transformer(x, missing_embed)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        return x



def build_model(state_dict: dict):
    from .vision_transformer_clip import CLIP  # 原 clip.py 不变

    vit = "visual.proj" in state_dict
    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
    )

    model.visual = VisionTransformer(
        input_resolution=image_resolution,
        patch_size=vision_patch_size,
        width=vision_width,
        layers=vision_layers,
        heads=vision_width // 64,
        output_dim=embed_dim,
        mamol_enable=True,
        mamol_cfg=dict(rank=8, num_dyn=2, num_stat=3, topk=1, router_dim=8),
        # mamol_layers=[10,11], 
        mamol_layers=[6, 7, 8, 9, 10, 11],  
        # mamol_layers=[],  

    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    try:
        model.load_state_dict(state_dict)
    except:
        missing, _ = model.load_state_dict(state_dict, strict=False)
        print("[Warning] Partial weights loaded:", missing)

    return model
