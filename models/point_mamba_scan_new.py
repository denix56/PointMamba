from typing import Union, Optional
import math
import random
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath

from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from mamba_ssm.modules.mamba_simple import Mamba
from .block_scan import Block

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from knn_cuda import KNN
from .build import MODELS
from .serialization import Point


class Encoder(nn.Module):  ## Embedding module
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # BG 512 n
        feature = self.second_conv(feature)  # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = misc.fps(xyz, self.num_group)  # B G 3
        # knn to get the neighborhood
        # idx = knn_query(xyz, center, self.group_size)  # B G M
        _, idx = self.knn(xyz, center)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


# ────────────────────────────────
# Local-Norm-Pool helper
# k-NN  ➜  L2-norm  ➜  2-layer MLP
# ────────────────────────────────
class LocalNormPool(nn.Module):
    def __init__(self, k: int = 16, in_c: int = 128, out_c: int = 128):
        super().__init__()
        self.k = k
        self.proj = nn.Sequential(
            nn.Linear(in_c, out_c),
            nn.GELU(),
            nn.Linear(out_c, out_c),
        )

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """
        xyz  : [B, G, 3]   group centres
        feat : [B, G, C]   token embeddings
        returns pooled features [B, G, C]
        """
        # k-nearest centres for each centre
        idx = torch.cdist(xyz, xyz).topk(self.k, largest=False).indices  # [B,G,k]
        B, G, C = feat.shape
        k = idx.size(-1)
        batch_idx = torch.arange(B, device=feat.device).view(B, 1, 1).expand(-1, G, k)
        neigh = feat[batch_idx, idx]  # [B,G,k,C]
        # ----------------------------------------------------------------

        pooled = neigh.norm(dim=-2)  # L2 over k
        return self.proj(pooled)


# ──────────────────────────────────────────────────────────────
# Local Geometric Pooling (learnable-k)  &  CoFE-SE definitions
# ──────────────────────────────────────────────────────────────
class LGP(nn.Module):
    def __init__(self, C, k_init=16):
        super().__init__()
        self.logk = nn.Parameter(torch.log(torch.tensor(float(k_init))))
        self.proj = nn.Sequential(
            nn.Linear(C + 1, C), nn.GELU(), nn.Linear(C, C)
        )

    def forward(self, xyz, feat):                     # xyz:[B,N,3]  feat:[B,N,C]
        k  = int(torch.clamp(self.logk.exp(), 4, 32).round().item())
        idx = torch.cdist(xyz, xyz).topk(k, largest=False).indices     # [B,N,k]

        B, N, C = feat.shape
        batch = torch.arange(B, device=feat.device).view(B, 1, 1).expand(-1, N, k)

        neigh_f = feat[batch, idx]                                       # [B,N,k,C]
        neigh_x = xyz[batch, idx]                                        # [B,N,k,3]

        delta_f = F.layer_norm(neigh_f - feat.unsqueeze(2), normalized_shape=(feat.size(-1),))
        offset  = neigh_x - neigh_x.mean(2, keepdim=True)
        sigma   = offset.std(2, keepdim=True) + 1e-6
        dist    = (offset / sigma).norm(dim=-1, keepdim=True)            # [B,N,k,1]
        w       = torch.exp(-0.5 * dist)

        fused   = torch.cat([delta_f * w, dist], -1).mean(2)             # [B,N,C+1]
        return self.proj(fused)                                          # [B,N,C]


class CoFE_SE(nn.Module):
    def __init__(self, C, groups=4):
        super().__init__()
        self.dw = nn.Conv1d(C, C, 3, padding=1, groups=C)
        self.norm = nn.GroupNorm(groups, C)
        self.se   = nn.Sequential(nn.Linear(C, C // 4), nn.GELU(),
                                  nn.Linear(C // 4, C), nn.Sigmoid())
        self.mix  = nn.Linear(C, C)

    def forward(self, x):                              # x [B,N,C]
        a = F.gelu(self.dw(x.transpose(1, 2))).transpose(1, 2)
        g = self.se(x.mean(1)).unsqueeze(1)            # [B,1,C]
        w = F.softmax((a + x * g).mean(-1, keepdim=True), dim=1)
        return self.mix(x * w)


# ──────────────────────────────────────────────────────────────
# Hybrid-Emba3D Mixer (LGP → Bi-Mamba → CoFE-SE)
# ──────────────────────────────────────────────────────────────
class EmbaMambaLayer(nn.Module):
    """
    Wraps one Mamba SSM block with:
        1. Local-Geometric-Pooling  (LGP)
        2. Bidirectional Mamba + Hadamard fusion
        3. CoFE-SE re-weighting
    """
    def __init__(self, d_model, layer_idx, k_init=16, groups=4, **kw):
        super().__init__()
        from mamba_ssm.modules.mamba_simple import Mamba          # local import
        self.lgp   = LGP(d_model, k_init)
        self.mamba = Mamba(d_model, layer_idx=layer_idx, **kw)
        self.cofe  = CoFE_SE(d_model, groups)

    def forward(self, feat, xyz, inference_params=None):         # feat:[B,N,C]
        # 1. LGP
        feat = feat + self.lgp(xyz, feat)

        # 2. Bidirectional Mamba + Hadamard
        y_fwd = self.mamba(feat, inference_params=inference_params)
        y_rev_tmp = self.mamba(torch.flip(feat, [1]), inference_params=inference_params)
        y_rev = torch.flip(y_rev_tmp, [1])
        feat = feat + y_fwd + y_rev + y_fwd * y_rev                # fusion

        # 3. CoFE-SE
        feat = feat + self.cofe(feat)
        return feat

# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def serialization(pos, feat=None, x_res=None, order="z", layers_outputs=[], grid_size=0.02):
    bs, n_p, _ = pos.size()
    if not isinstance(order, list):
        order = [order]

    scaled_coord = pos / grid_size
    grid_coord = torch.floor(scaled_coord).to(torch.int64)
    min_coord = grid_coord.min(dim=1, keepdim=True)[0]
    grid_coord = grid_coord - min_coord

    batch_idx = torch.arange(0, pos.shape[0], 1.0).unsqueeze(1).repeat(1, pos.shape[1]).to(torch.int64).to(pos.device)

    point_dict = {'batch': batch_idx.flatten(), 'grid_coord': grid_coord.flatten(0, 1), }
    point_dict = Point(**point_dict)
    point_dict.serialization(order=order)

    order = point_dict.serialized_order
    inverse_order = point_dict.serialized_inverse

    pos = pos.flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()
    if feat is not None:
        feat = feat.flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()
    if x_res is not None:
        x_res = x_res.flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()

    for i in range(len(layers_outputs)):
        layers_outputs[i] = layers_outputs[i].flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()
    return pos, order, inverse_order, feat, x_res


def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        drop_path=0.,
        device=None,
        dtype=None,
):

    if ssm_cfg is None:
        ssm_cfg = {}
    else:
        ssm_cfg = ssm_cfg.copy()
    factory_kwargs = {"device": device, "dtype": dtype}

    emba = ssm_cfg.pop("emba", False)  # ← YAML flag
    if emba:
        mixer_cls = partial(EmbaMambaLayer, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    else:
        mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        drop_path=drop_path,
    )
    block.layer_idx = layer_idx
    return block


class MixerModel(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            ssm_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            drop_out: int = 0.,
            drop_path=0.,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.drop_out = nn.Dropout(drop_out) if drop_out > 0. else nn.Identity()

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, pos, xyz=None, inference_params=None):
        hidden_states = input_ids + pos
        for layer in self.layers:
            # EmbaMambaLayer expects xyz; plain Mamba ignores it
            hidden_states = layer(
                hidden_states,
                xyz=xyz if xyz is not None else hidden_states.new_zeros(
                    hidden_states.shape[0], hidden_states.shape[1], 3
                ),
                inference_params=inference_params,
            )
            hidden_states = self.drop_out(hidden_states)
        return self.norm_f(hidden_states.to(dtype=self.norm_f.weight.dtype))


@MODELS.register_module()
class PointMambaScan(nn.Module):
    def __init__(self, config, **kwargs):
        super(PointMambaScan, self).__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.cls_dim = config.cls_dim

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        # Local-Norm-Pool (HiP-Mamba enhancement)
        self.local_pool = LocalNormPool(
            k=getattr(config, "local_k", 16),  # YAML or default 16
            in_c=self.encoder_dims,
            out_c=self.encoder_dims,  # keep feature dim unchanged
        )

        self.use_cls_token = False if not hasattr(self.config, "use_cls_token") else self.config.use_cls_token
        drop_path = 0. if not hasattr(self.config, "drop_path") else self.config.drop_path
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.depth)]
        self.rms_norm = False if not hasattr(self.config, "rms_norm") else self.config.rms_norm
        self.drop_out = 0. if not hasattr(self.config, "drop_out") else self.config.drop_out
        self.max_head = False if not hasattr(self.config, "max_head") else self.config.max_head
        self.avg_head = False if not hasattr(self.config, "avg_head") else self.config.avg_head

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
            self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
            trunc_normal_(self.cls_token, std=.02)
            trunc_normal_(self.cls_pos, std=.02)

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.blocks = MixerModel(d_model=self.trans_dim,
                                 n_layer=self.depth,
                                 rms_norm=self.rms_norm,
                                 drop_out=self.drop_out,
                                 drop_path=dpr)

        self.HEAD_CHANEL = 1
        if self.use_cls_token:
            self.HEAD_CHANEL += 1
        if self.max_head and self.avg_head:
            self.HEAD_CHANEL += 1

        self.cls_head_finetune = nn.Sequential(
            nn.Linear(self.trans_dim * self.HEAD_CHANEL, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.cls_dim)
        )

        self.build_loss_func()

        self.OrderScale_gamma_1, self.OrderScale_beta_1 = init_OrderScale(self.trans_dim)
        self.OrderScale_gamma_2, self.OrderScale_beta_2 = init_OrderScale(self.trans_dim)

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path, map_location='cpu')
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                # if 'cls_head_finetune' in k:
                #     del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Mamba')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Mamba'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Mamba')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Mamba'
                )

            print_log(f'[Mamba] Successful Loading the ckpt from {bert_ckpt_path}', logger='Mamba')
        else:
            print_log('Training from scratch!!!', logger='Mamba')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts):
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N
        # enrich each group token with its k-NN context
        group_input_tokens = self.local_pool(center, group_input_tokens)
        pos = self.pos_embed(center)  # B G C

        # # reordering strategy
        _, _, _, group_input_tokens_forward, pos_forward = serialization_func(center, group_input_tokens, pos,
                                                                              'hilbert')
        _, _, _, group_input_tokens_backward, pos_backward = serialization_func(center, group_input_tokens, pos,
                                                                                'hilbert-trans')
        group_input_tokens_forward = apply_OrderScale(group_input_tokens_forward,
                                                      self.OrderScale_gamma_1, self.OrderScale_beta_1)
        group_input_tokens_backward = apply_OrderScale(group_input_tokens_backward,
                                                       self.OrderScale_gamma_2, self.OrderScale_beta_2)
        if self.use_cls_token:
            cls_token = self.cls_token.expand(group_input_tokens_forward.size(0), -1, -1)
            cls_pos = self.cls_pos.expand(group_input_tokens_forward.size(0), -1, -1)
            pos = torch.cat([pos_forward, pos_backward, cls_token], dim=1)
            group_input_tokens = torch.cat([group_input_tokens_forward, group_input_tokens_backward, cls_pos], dim=1)
        else:
            pos = torch.cat([pos_forward, pos_backward], dim=1)
            group_input_tokens = torch.cat([group_input_tokens_forward, group_input_tokens_backward], dim=1)

        # cls_token = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        # cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        # pos = torch.cat([cls_pos, pos], dim=1)
        # group_input_tokens = torch.cat([cls_token, group_input_tokens], dim=1)
        #
        # # group_input_tokens = group_input_tokens_forward
        # # pos = pos_forward

        x = group_input_tokens
        x = self.blocks(x, pos)
        # x = self.norm(x)
        if self.use_cls_token:
            cls_token = x[:, -1, :]
            concat_f = torch.cat([cls_token, x.mean(1)], dim=1)
        else:
            concat_f = None
            if self.avg_head:
                if concat_f is None:
                    concat_f = x[:, :].mean(1)
                else:
                    concat_f = torch.cat([concat_f, x[:, :].mean(1)], dim=1)
            if self.max_head:
                if concat_f is None:
                    concat_f = x[:, :].max(1)[0]
                    # concat_f = x[:, -1]
                else:
                    concat_f = torch.cat([concat_f, x[:, :].max(1)[0]], dim=1)
                    # concat_f = torch.cat([concat_f, x[:, -1]], dim=1)
            if concat_f is None:
                assert False
        # concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)

        ret = self.cls_head_finetune(concat_f)
        return ret


class MaskMamba(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.mask_ratio = config.mamba_config.mask_ratio
        self.trans_dim = config.mamba_config.trans_dim
        self.depth = config.mamba_config.depth
        self.drop_path_rate = config.mamba_config.drop_path_rate
        print_log(f'[args] {config.mamba_config}', logger='Mamba')
        # embedding
        self.encoder_dims = config.mamba_config.encoder_dims
        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        # Local-Norm-Pool (HiP-Mamba enhancement)
        self.local_pool = LocalNormPool(
            k=getattr(config, "local_k", 16),  # YAML or default 16
            in_c=self.encoder_dims,
            out_c=self.encoder_dims,  # keep feature dim unchanged
        )

        self.mask_type = config.mamba_config.mask_type
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]

        self.blocks = MixerModel(d_model=self.trans_dim,
                                 n_layer=self.depth,
                                 rms_norm=self.config.rms_norm,
                                 drop_path=dpr,
                                 ssm_cfg={'emba': config.mamba_config.emba}
                                 )

        self.OrderScale_gamma_1, self.OrderScale_beta_1 = init_OrderScale(self.trans_dim)
        self.OrderScale_gamma_2, self.OrderScale_beta_2 = init_OrderScale(self.trans_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _mask_center_block(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0)  # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2,
                                         dim=-1)  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G
            ratio = self.mask_ratio
            mask_num = int(ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G - self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(center.device)  # B G

    def forward(self, neighborhood, center, order, order_index, noaug=False):
        # ---------- 1. build mask (unchanged) ----------
        if self.mask_type == "rand":
            bool_masked_pos = self._mask_center_rand(center, noaug=noaug)
        else:
            bool_masked_pos = self._mask_center_block(center, noaug=noaug)

        # ---------- 2. encode & local pool ----------
        group_input_tokens = self.encoder(neighborhood)  # [B,G,C]
        group_input_tokens = self.local_pool(center, group_input_tokens)

        # ---------- 3. OrderScale (unchanged) ----------
        if order_index == 0:
            group_input_tokens = apply_OrderScale(
                group_input_tokens, self.OrderScale_gamma_1, self.OrderScale_beta_1
            )
        elif order_index == 1:
            group_input_tokens = apply_OrderScale(
                group_input_tokens, self.OrderScale_gamma_2, self.OrderScale_beta_2
            )
        else:
            raise ValueError("order_index must be 0 or 1")

        # ---------- 4. slice visible tokens & coords ----------
        B, G, C = group_input_tokens.shape
        x_vis = group_input_tokens[~bool_masked_pos].reshape(B, -1, C)  # [B,V,C]
        xyz_vis = center[~bool_masked_pos].reshape(B, -1, 3)  # [B,V,3]
        pos = self.pos_embed(xyz_vis)  # [B,V,C]

        # ---------- 5. run through Emba-Mamba stack ----------
        x_vis = self.blocks(x_vis, pos, xyz_vis)  # ← note extra xyz_vis arg

        # ---------- 6. return (unchanged signature) ----------
        return x_vis, bool_masked_pos, group_input_tokens


class MambaDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, norm_layer=nn.LayerNorm, drop_path=0, config=None):
        super().__init__()

        self.blocks = MixerModel(d_model=embed_dim,
                                 n_layer=depth,
                                 rms_norm=config.rms_norm,
                                 drop_path=drop_path,
                                 ssm_cfg={'emba': config.mamba_config.emba})
        self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, xyz, mask_pos):
        """ x/pos/xyz: full tokens   mask_pos: bool mask for masked centres """
        x = self.blocks(x, pos, xyz)              # xyz forwarded to each layer
        x = self.head(x[mask_pos, :].view(x.size(0), -1, x.size(-1)))
        return x

    # def forward(self, x, pos, N):
    #     B, _, C = x.shape
    #     x = self.blocks(x, pos)
    #
    #     x = self.head(x[:, -N:, :])  # only return the mask tokens predict pixel
    #     return x


@MODELS.register_module()
class Point_MAE_Mamba_serializationV2(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[PointMamba] ', logger='PointMamba')
        self.config = config
        self.trans_dim = config.mamba_config.trans_dim
        self.MAE_encoder = MaskMamba(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.decoder_depth = config.mamba_config.decoder_depth
        self.drop_path_rate = config.mamba_config.drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]

        self.MAE_decoder = MambaDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path=dpr,
            config=config,
        )

        print_log(f'[PointMamba] divide point cloud into G{self.num_group} x S{self.group_size} points ...',
                  logger='PointMamba')
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        # prediction head
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 3 * self.group_size, 1)
        )

        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type == 'cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError
            # self.loss_func = emd().cuda()

    def forward(self, pts, vis=False, **kwargs):
        neighborhood, center = self.group_divider(pts)
        B, G, S, _ = neighborhood.shape

        order_list = ['hilbert', 'hilbert-trans']
        # order_list = ['hilbert']
        order_index = np.random.choice([i for i in range(len(order_list))], 1, replace=False)

        center, order, index_order, _, _ = serialization_func(center, None, None, order_list[order_index[0]])
        neighborhood = neighborhood.flatten(0, 1)[order].reshape(B, G, S, -1).contiguous()

        x_vis, mask, group_input_tokens = self.MAE_encoder(neighborhood, center, order=order,
                                                           order_index=order_index[0])
        B, _, C = x_vis.shape  # B VIS C

        pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)

        pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)

        x_full = torch.zeros_like(group_input_tokens, device=group_input_tokens.device)
        x_full[~mask] = x_vis.reshape(-1, C)
        x_full[mask] = self.mask_token.reshape(-1, C)
        pos_full = torch.zeros_like(group_input_tokens, device=group_input_tokens.device)
        pos_full[~mask] = pos_emd_vis.reshape(-1, C).to(pos_full.dtype)
        pos_full[mask] = pos_emd_mask.reshape(-1, C).to(pos_full.dtype)

        x_rec = self.MAE_decoder(x_full, pos_full, center, mask)

        B, M, C = x_rec.shape
        rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024

        gt_points = neighborhood[mask].reshape(B * M, -1, 3)
        loss1 = self.loss_func(rebuild_points.to(torch.float32), gt_points.to(torch.float32))

        if vis:  # visualization
            vis_points = neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)
            full_vis = vis_points + center[~mask].unsqueeze(1)
            full_rebuild = rebuild_points + center[mask].unsqueeze(1)
            full = torch.cat([full_vis, full_rebuild], dim=0)
            # full_points = torch.cat([rebuild_points,vis_points], dim=0)
            full_center = torch.cat([center[mask], center[~mask]], dim=0)
            # full = full_points + full_center.unsqueeze(1)
            ret2 = full_vis.reshape(-1, 3).unsqueeze(0)
            ret1 = full.reshape(-1, 3).unsqueeze(0)
            # return ret1, ret2
            return ret1, ret2, full_center
        else:
            return loss1


def init_OrderScale(dim):
    gamma = nn.Parameter(torch.ones(dim))
    beta = nn.Parameter(torch.zeros(dim))
    nn.init.normal_(gamma, mean=1, std=.02)
    nn.init.normal_(beta, std=.02)
    return gamma, beta


def apply_OrderScale(x, gamma, beta):
    assert gamma.shape == beta.shape
    if x.shape[-1] == gamma.shape[0]:
        return x * gamma + beta
    elif x.shape[1] == gamma.shape[0]:
        return x * gamma.view(1, -1, 1, 1) + beta.view(1, -1, 1, 1)
    else:
        raise ValueError('the input tensor shape does not match the shape of the scale factor.')


def serialization_func(p, x, x_res, order, layers_outputs=[]):
    p, order, inverse_order, x, x_res = serialization(p, x, x_res=x_res, order=order,
                                                      layers_outputs=layers_outputs,
                                                      grid_size=0.02)
    return p, order, inverse_order, x, x_res
