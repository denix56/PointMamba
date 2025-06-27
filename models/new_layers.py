import torch
from torch import nn

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from .utils import MixerModel


class StochasticNeuralSortPermuter(nn.Module):
    def __init__(self, tau_2: float = 4.0):
        super().__init__()
        self.tau_2 = tau_2

    def forward(self, z: torch.Tensor, tau: float, hard: bool = True) -> torch.Tensor:
        """
        z: [B, N]  — learned log-scores for each patch
        returns:
          P: [B, N, N]  — a soft permutation matrix, stochastically sampled
        """
        B, dim = z.shape
        eps = torch.finfo(z.dtype).eps
        # 1) sample Gumbel noise
        g = -torch.log(-torch.log(torch.rand_like(z) + eps) + eps)
        z_tilde = z + tau * g  # [B, N]

        # 2) build pairwise absolute-diff matrix
        #    shape [B, N, N]: |z_tilde_i - z_tilde_j|
        A = torch.abs(z_tilde.unsqueeze(2) - z_tilde.unsqueeze(1))
        B = A.sum(dim=2, keepdim=True).expand(-1, -1, A.size(1))
        scaling = (dim + 1 - 2 * torch.arange(1, dim + 1, device=z.device, dtype=z.dtype))
        C = torch.matmul(z_tilde.unsqueeze(-1), scaling.unsqueeze(0))

        P_max = (C - B).transpose(1, 2)
        P_hat = torch.softmax(P_max / self.tau_2, dim=-1)
        if hard:
            P_hard = torch.zeros_like(P_hat).scatter(-1, P_hat.argmax(dim=-1, keepdim=True), 1.0)
            P_hat = P_hard + (P_hat - P_hat.detach())
        return P_hat


class CouplingLayer(nn.Module):
    def __init__(self, d_model,
                 n_layer: int,
                 ssm_cfg=None,
                 norm_epsilon=1e-5,
                 rms_norm=False,
                 residual_in_fp32=False,
                 fused_add_norm=False,
                 drop_path=0.):
        super().__init__()

        assert d_model % 2 == 0

        self.mixer = MixerModel(d_model // 2, n_layer=n_layer, ssm_cfg=ssm_cfg,
                                  norm_epsilon=norm_epsilon,
                                  rms_norm=rms_norm,
                                  residual_in_fp32=residual_in_fp32,
                                  fused_add_norm=fused_add_norm,
                                  drop_path=drop_path)

        self.scale = nn.Linear(d_model // 2, d_model // 2)
        self.bias = nn.Linear(d_model // 2, d_model // 2)
        self.m = 2

    def forward(self, x: torch.Tensor, inference_params=None) -> torch.Tensor:
        assert x.size(-1) % 2 == 0
        x1, x2 = x.split(2, dim=-1)
        h1 = self.mixer(x1, inference_params=inference_params)
        y2 = (x2 + self.bias(h1)) * torch.sigmoid(self.scale(h1)) * self.m
        y  = torch.cat((x1, y2), dim=-1)
        return y


class JetFlow(nn.Module):
    def __init__(self, d_model,
                 n_couple_layer: int,
                 n_layer: int,
                 ssm_cfg=None,
                 norm_epsilon=1e-5,
                 rms_norm=False,
                 residual_in_fp32=False,
                 fused_add_norm=False,
                 drop_path=0.):
        super().__init__()
        layers = []
        for i in range(n_couple_layer):
            layers.append(CouplingLayer(d_model=d_model, n_layer=n_layer,ssm_cfg=ssm_cfg,norm_epsilon=norm_epsilon,
                                        rms_norm=rms_norm, residual_in_fp32=residual_in_fp32,
                                        fused_add_norm=fused_add_norm,drop_path=drop_path))
        self.layers = nn.ModuleList(layers)
