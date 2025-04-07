from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List
from timm.layers.helpers import to_2tuple
import pdb 

def normalized_adj(adj):
    # caculate the degree of each node, fill 0 with 1 to avoid divide by zero
    degree = torch.sum(adj, dim=1)
    degree[degree < 1e-5] = 1
    return torch.bmm(torch.bmm(torch.diag_embed(torch.pow(degree, -0.5)), adj), torch.diag_embed(torch.pow(degree, -0.5)))

def softmax(x: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
    """
    Compute the softmax along the specified dimensions.
    This function adds the option to specify multiple dimensions

    Args:
        x (torch.Tensor): Input tensor.
        dims (int or tuple[int]): The dimension or list of dimensions along which the softmax probabilities are computed.

    Returns:
        torch.Tensor: Output tensor containing softmax probabilities along the specified dimensions.
    """
    max_vals = torch.amax(x, dim=dim, keepdim=True)
    e_x = torch.exp(x - max_vals)
    sum_exp = e_x.sum(dim=dim, keepdim=True)
    return e_x / sum_exp


class SoftMoELayerWrapper(nn.Module):
    """
    A wrapper class to create a Soft Mixture of Experts layer.

    From "From Sparse to Soft Mixtures of Experts"
    https://arxiv.org/pdf/2308.00951.pdf
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        slots_per_expert: int,
        layer: Callable,
        normalize: bool = True,
        **layer_kwargs,
    ) -> None:
        """
        Args:
            dim (int): Dimensionality of input features.
            num_experts (int): Number of experts.
            slots_per_expert (int): Number of token slots per expert.
            layer (Callable): Network layer of the experts.
            normalize (bool): Normalize input and phi (sec. 2.3 from paper)
            **layer_kwargs: Additional keyword arguments for the layer class.
        """
        super().__init__()

        self.dim = dim
        self.num_experts = num_experts
        self.slots_per_expert = slots_per_expert
        self.normalize = normalize

        # Initialize phi and normalization scaling factor
        self.phi = nn.Parameter(torch.zeros(dim, num_experts, slots_per_expert))
        if self.normalize:
            self.scale = nn.Parameter(torch.ones(1))

        # Initialize phi using LeCun normal initialization
        # https://github.com/google-research/vmoe/blob/662341d007650d5bbb7c6a2bef7f3c759a20cc7e/vmoe/projects/soft_moe/router.py#L49C1-L49C1
        nn.init.normal_(self.phi, mean=0, std=1 / dim**0.5)

        # Create a list of expert networks
        self.experts = nn.ModuleList(
            [layer(**layer_kwargs) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Soft-MoE layer (algorithm 1 from paper).

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, input_dim].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, input_dim].
        """
        assert (
            x.shape[-1] == self.dim
        ), f"Input feature dim of {x.shape[-1]} does not match layer dim of {self.dim}"
        assert (
            len(x.shape) == 3
        ), f"Input expected to have 3 dimensions but has {len(x.shape)}"

        phi = self.phi

        # Normalize input and phi
        if self.normalize:
            x = F.normalize(x, dim=2)  # [b, m, d]
            phi = self.scale * F.normalize(phi, dim=0)  # [d, n, p]

        # Compute dispatch and combine weights
        logits = torch.einsum("bmd,dnp->bmnp", x, phi)
        d = softmax(logits, dim=1)
        c = softmax(logits, dim=(2, 3))

        # Compute input slots as weighted average of input tokens using dispatch weights
        xs = torch.einsum("bmd,bmnp->bnpd", x, d)

        # Apply expert to corresponding slots
        ys = torch.stack(
            [f_i(xs[:, i, :, :]) for i, f_i in enumerate(self.experts)], dim=1
        )

        # Compute output tokens as weighted average of output slots using combine weights
        y = torch.einsum("bnpd,bmnp->bmd", ys, c)

        return y

class Soft_GRAPH_MoELayerWrapper(nn.Module):
    """
    A wrapper class to create a Soft Mixture of Experts layer.

    From "From Sparse to Soft Mixtures of Experts"
    https://arxiv.org/pdf/2308.00951.pdf
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        slots_per_expert: int,
        layer: Callable,
        beta: float, 
        alpha: float, 
        normalize: bool = True,
        **layer_kwargs,
    ) -> None:
        """
        Args:
            dim (int): Dimensionality of input features.
            num_experts (int): Number of experts.
            slots_per_expert (int): Number of token slots per expert.
            layer (Callable): Network layer of the experts.
            normalize (bool): Normalize input and phi (sec. 2.3 from paper)
            **layer_kwargs: Additional keyword arguments for the layer class.
        """
        super().__init__()

        self.dim = dim
        self.num_experts = num_experts
        self.slots_per_expert = slots_per_expert
        self.normalize = normalize

        # Initialize phi and normalization scaling factor
        self.phi = nn.Parameter(torch.zeros(dim, num_experts, slots_per_expert))

        self.register_buffer('adj', torch.zeros((slots_per_expert, self.num_experts,self.num_experts)))
        self.register_buffer('new_adj', torch.zeros((slots_per_expert, self.num_experts,self.num_experts)))
        self.adj_norm = normalized_adj
        self.beta = beta
        self.alpha = alpha
        self.warm_up_over = False

        if self.normalize:
            self.scale = nn.Parameter(torch.ones(1))

        # Initialize phi using LeCun normal initialization
        # https://github.com/google-research/vmoe/blob/662341d007650d5bbb7c6a2bef7f3c759a20cc7e/vmoe/projects/soft_moe/router.py#L49C1-L49C1
        nn.init.normal_(self.phi, mean=0, std=1 / dim**0.5)

        # Create a list of expert networks
        self.experts = nn.ModuleList(
            [layer(**layer_kwargs) for _ in range(num_experts)]
        )

    def update_adj(self):
        """Method to update the adjacency matrix, will be called at the end of each epoch
        """
        # first time update adj matrix
        print(">>>>>>>>>> updating adj <<<<<<<<<<<<")
        if self.adj.sum() <= 1e-3:
            self.adj = self.adj_norm(self.new_adj)
        else:
            self.adj = self.beta * self.adj + (1-self.beta) * self.adj_norm(self.new_adj)
        print(self.adj)
        self.new_adj.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Soft-MoE layer (algorithm 1 from paper).

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, input_dim].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, input_dim].
        """
        assert (
            x.shape[-1] == self.dim
        ), f"Input feature dim of {x.shape[-1]} does not match layer dim of {self.dim}"
        assert (
            len(x.shape) == 3
        ), f"Input expected to have 3 dimensions but has {len(x.shape)}"

        phi = self.phi

        # Normalize input and phi
        if self.normalize:
            x = F.normalize(x, dim=2)  # [b, m, d]
            phi = self.scale * F.normalize(phi, dim=0)  # [d, n, p]

        if self.training:
            with torch.no_grad():
                # [p, n, n]
                self.new_adj += torch.bmm(phi.permute(2, 1, 0), phi.permute(2, 0, 1))


        if not self.adj.sum() <= 1e-3 and self.warm_up_over:
            full_adj = self.alpha * torch.eye(self.num_experts, device=self.phi.device).unsqueeze(0) + (1-self.alpha) * self.adj
            # [p, d, n]
            phi = torch.bmm(phi.permute(2, 0, 1), torch.transpose(full_adj, -1, -2)).permute(1, 2, 0)

        # Compute dispatch and combine weights
        logits = torch.einsum("bmd,dnp->bmnp", x, phi)
        d = softmax(logits, dim=1)
        c = softmax(logits, dim=(2, 3))

        # Compute input slots as weighted average of input tokens using dispatch weights
        xs = torch.einsum("bmd,bmnp->bnpd", x, d)

        # Apply expert to corresponding slots
        ys = torch.stack(
            [f_i(xs[:, i, :, :]) for i, f_i in enumerate(self.experts)], dim=1
        )

        # Compute output tokens as weighted average of output slots using combine weights
        y = torch.einsum("bnpd,bmnp->bmd", ys, c)

        return y

class Soft_GRAPH_threshold_MoELayerWrapper(nn.Module):
    """
    A wrapper class to create a Soft Mixture of Experts layer.

    From "From Sparse to Soft Mixtures of Experts"
    https://arxiv.org/pdf/2308.00951.pdf
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        slots_per_expert: int,
        layer: Callable,
        beta: float, 
        alpha: float, 
        t: float,
        normalize: bool = True,
        **layer_kwargs,
    ) -> None:
        """
        Args:
            dim (int): Dimensionality of input features.
            num_experts (int): Number of experts.
            slots_per_expert (int): Number of token slots per expert.
            layer (Callable): Network layer of the experts.
            normalize (bool): Normalize input and phi (sec. 2.3 from paper)
            **layer_kwargs: Additional keyword arguments for the layer class.
        """
        super().__init__()

        self.dim = dim
        self.num_experts = num_experts
        self.slots_per_expert = slots_per_expert
        self.normalize = normalize

        # Initialize phi and normalization scaling factor
        self.phi = nn.Parameter(torch.zeros(dim, num_experts, slots_per_expert))

        self.register_buffer('adj', torch.zeros((slots_per_expert*self.num_experts,slots_per_expert*self.num_experts)))
        self.register_buffer('new_adj', torch.zeros((slots_per_expert*self.num_experts,slots_per_expert*self.num_experts)))
        self.adj_norm = normalized_adj
        self.beta = beta
        self.alpha = alpha
        self.t = t
        self.warm_up_over = False

        if self.normalize:
            self.scale = nn.Parameter(torch.ones(1))

        # Initialize phi using LeCun normal initialization
        # https://github.com/google-research/vmoe/blob/662341d007650d5bbb7c6a2bef7f3c759a20cc7e/vmoe/projects/soft_moe/router.py#L49C1-L49C1
        nn.init.normal_(self.phi, mean=0, std=1 / dim**0.5)

        # Create a list of expert networks
        self.experts = nn.ModuleList(
            [layer(**layer_kwargs) for _ in range(num_experts)]
        )

    def update_adj(self):
        """Method to update the adjacency matrix, will be called at the end of each epoch
        """
        # first time update adj matrix
        print(">>>>>>>>>> updating adj <<<<<<<<<<<<")
        if self.adj.sum() <= 1e-3:
            self.adj = self.adj_norm(self.new_adj)
        else:
            self.adj = self.beta * self.adj + (1-self.beta) * self.adj_norm(self.new_adj)
        print(self.adj)
        self.new_adj.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Soft-MoE layer (algorithm 1 from paper).

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, input_dim].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, input_dim].
        """
        assert (
            x.shape[-1] == self.dim
        ), f"Input feature dim of {x.shape[-1]} does not match layer dim of {self.dim}"
        assert (
            len(x.shape) == 3
        ), f"Input expected to have 3 dimensions but has {len(x.shape)}"

        phi = self.phi

        # Normalize input and phi
        if self.normalize:
            x = F.normalize(x, dim=2)  # [b, m, d]
            phi = self.scale * F.normalize(phi, dim=0)  # [d, n, p]

        # Compute dispatch and combine weights
        logits = torch.einsum("bmd,dnp->bmnp", x, phi)
        # this is [b, m, n, p]
        d = softmax(logits, dim=1)
        b, m, n, p = d.shape

        if self.training:
            with torch.no_grad():
                d_ = d.reshape(b, m, n*p)
                # [b, np, np]
                batch_adj = torch.abs(F.cosine_similarity(d_, d_))
                batch_adj[batch_adj >= self.t] = 1
                batch_adj[batch_adj < self.t] = 0
                self.new_adj += batch_adj.sum(dim=0)

        if not self.adj.sum() <= 1e-3 and self.warm_up_over:
            # [p, n, n]
            full_adj = self.alpha * torch.eye(self.num_experts*self.slots_per_expert, device=self.phi.device) + (1-self.alpha) * self.adj
            # [p, d, n]
            d = d.reshape(b, m, n*p)
            d = torch.bmm(d, full_adj).reshape(b, m, n, p)

        c = softmax(logits, dim=(2, 3))

        # Compute input slots as weighted average of input tokens using dispatch weights
        xs = torch.einsum("bmd,bmnp->bnpd", x, d)

        # Apply expert to corresponding slots
        ys = torch.stack(
            [f_i(xs[:, i, :, :]) for i, f_i in enumerate(self.experts)], dim=1
        )

        # Compute output tokens as weighted average of output slots using combine weights
        y = torch.einsum("bnpd,bmnp->bmd", ys, c)

        return y

class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=False, attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # sqrt (D)
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn) 

        # @ is a matrix multiplication
        x = (attn @ v)
        x = x.transpose(1, 2).reshape(B,N,C)
       
        x = self.proj(x)
        x = self.proj_drop(x)
       
        return x

class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x