import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy
from statistics import mean
import numpy as np
 
import torch
import torch.nn as nn
import torch.nn.functional as F
 
from timm.models.layers import PatchEmbed, DropPath, trunc_normal_, lecun_normal_
from timm.models.vision_transformer import init_weights_vit_timm, _load_weights, init_weights_vit_jax, get_init_weights_vit
from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv
from utils import named_apply
import copy
from typing import Callable

import torch
import torch.jit
import torch.nn as nn
import torch.utils.checkpoint
from timm.layers import PatchDropout, PatchEmbed, trunc_normal_
from timm.models._manipulate import checkpoint_seq, named_apply
from softmoe import Block, Mlp
from softmoe import SoftMoELayerWrapper, Soft_GRAPH_MoELayerWrapper, Soft_GRAPH_threshold_MoELayerWrapper


 
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., layerth=0,ttl_tokens=0):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.layerth = layerth
        self.ttl_tokens = ttl_tokens
        self.s = nn.Parameter(torch.zeros(self.num_heads, self.ttl_tokens, self.ttl_tokens))
        head_dim = dim // num_heads
        # sqrt (D)
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.v = torch.zeros((1,1,dim))
        self.q = torch.zeros((1,1,dim))
        self.k = torch.zeros((1,1,dim))
        # self.x_in = np.zeros((1,1,dim))
        # self.x_out = np.zeros((1,1,dim))
        
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # self.x_in = x.cpu().numpy()
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        self.v = v.permute(0,2,1,3).reshape(B,N,C).cpu()
        self.q = q.permute(0,2,1,3).reshape(B,N,C).cpu()
        self.k = k.permute(0,2,1,3).reshape(B,N,C).cpu()

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        s = self.s.unsqueeze(dim=0).expand(B,self.num_heads,N,N)
        I = torch.eye(N,N).unsqueeze(dim=0).unsqueeze(dim=0).expand(B,self.num_heads,N,N).to(torch.device("cuda"), non_blocking=True)
        v = (I-s) @ v

        attn = self.attn_drop(attn) 

        # @ is a matrix multiplication
        x = (attn @ v)
        x = x.transpose(1, 2).reshape(B,N,C)
       
        x = self.proj(x)
        x = self.proj_drop(x)
        # self.x_out = x.cpu().numpy()
        
        ################ COSINE SIMILARITY MEASURE
        # n = x.shape[1] #x is in shape of (batchsize, length, dim)
        # sqaure norm across features
        # x_norm = torch.norm(x, 2, dim = -1, keepdim= True)
        # x_ = x/x_norm
        # x_cossim = torch.tril((x_ @ x_.transpose(-2, -1)), diagonal= -1).sum(dim = (-1, -2))/(n*(n - 1)/2)
        # x_cossim = x_cossim.mean()
        # python debugger breakpoint
#         import pdb;pdb.set_trace()
        ################
       
        return x


# class Block(nn.Module):
 
#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
#                  drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, layerth = None,ttl_tokens=0):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
#                                     attn_drop=attn_drop, proj_drop=drop,layerth=layerth,ttl_tokens=ttl_tokens)
#         # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
#         self.layerth = layerth
 
#     def forward(self, x):

#         x = x + self.drop_path(self.attn(self.norm1(x)))
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#         return x
 
 
class VisionTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """
 
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='',pretrained_cfg=None,pretrained_cfg_overlay=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
       
        # how does embedding conv2d update its weights? 
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        # print(img_size,patch_size,in_chans,num_patches)
 
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        self.v = torch.zeros((1,num_patches+self.num_tokens,self.embed_dim))
        self.q = torch.zeros((1,num_patches+self.num_tokens,self.embed_dim))
        self.k = torch.zeros((1,num_patches+self.num_tokens,self.embed_dim))
        # self.x_in = np.zeros((1,num_patches+self.num_tokens,self.embed_dim))
        # self.x_out = np.zeros((1,num_patches+self.num_tokens,self.embed_dim))
 
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, layerth = i,
                ttl_tokens=num_patches+self.num_tokens)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
 
        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([f
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()
 
        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()
 
        self.init_weights(weight_init)
 
    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            partial(init_weights_vit_jax(mode, head_bias), head_bias=head_bias, jax_impl=True)
        else:
            trunc_normal_(self.cls_token, std=.02)
            init_weights_vit_timm
 
    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        init_weights(m)
 
    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)
 
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}
 
    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist
 
    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()
 
    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        # add the same pos_emb token to each sample? broadcasting...
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        self.v = torch.zeros((x.shape[0],x.shape[1],self.embed_dim))
        self.q = torch.zeros((x.shape[0],x.shape[1],self.embed_dim))
        self.k = torch.zeros((x.shape[0],x.shape[1],self.embed_dim))
        # self.x_in = np.zeros((x.shape[0],x.shape[1],self.embed_dim))
        # self.x_out = np.zeros((x.shape[0],x.shape[1],self.embed_dim))
        
        for name, block in self.blocks.named_children():
            if self.v.sum() == 0:
                self.v = block.attn.v.unsqueeze(dim=1)
            else:
                self.v = torch.cat((self.v,block.attn.v.unsqueeze(dim=1)),dim=1)
            if self.k.sum() == 0:
                self.k = block.attn.k.unsqueeze(dim=1)
            else:
                self.k = torch.cat((self.k,block.attn.k.unsqueeze(dim=1)),dim=1)
            if self.q.sum() == 0:
                self.q = block.attn.q.unsqueeze(dim=1)
            else:
                self.q = torch.cat((self.q,block.attn.q.unsqueeze(dim=1)),dim=1)
        print(self.v.shape)
        print(self.k.shape)
        print(self.q.shape)
        #     if self.x_in.flatten().sum() == 0:
        #         self.x_in = np.array([block.attn.x_in])
        #     else:
        #         self.x_in = np.append(self.x_in,[block.attn.x_in],axis=0)
        #     if self.x_out.flatten().sum() == 0:
        #         self.x_out = np.array([block.attn.x_out])
        #     else:
        #         self.x_out = np.append(self.x_out,[block.attn.x_out],axis=0)
                
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]
 
    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        # return x, self.v, self.q, self.k, self.x_in, self.x_out
        return x, self.v, self.q, self.k
        # return x


class SoftMoEVisionTransformer(nn.Module):
    """Vision Transformer with Soft Mixture of Experts MLP layers.

    From the paper "From Sparse to Soft Mixtures of Experts"
    https://arxiv.org/pdf/2308.00951.pdf

    Code modified from:
    https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
    """

    def __init__(
        self,
        num_experts: int = 128,
        slots_per_expert: int = 1,
        moe_layer_index: int | list[int] = 6,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: str = "token",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        init_values: float | None = None,
        class_token: bool = True,
        no_embed_class: bool = False,
        pre_norm: bool = False,
        fc_norm: bool | None = None,
        drop_rate: float = 0.0,
        pos_drop_rate: float = 0.0,
        patch_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        weight_init: str = "",
        embed_layer: Callable = PatchEmbed,
        norm_layer: Callable | None = None,
        act_layer: Callable | None = None,
        block_fn: Callable = Block,
        mlp_layer: Callable = Mlp,
        pretrained_cfg=None,
        pretrained_cfg_overlay=None,
    ):
        """
        Args:
            num_experts (int): Number of experts in MoE layers.
            slots_per_expert (int): Number of token slots per expert.
            moe_layer_index (int or list[int]): Block depth indices where MoE layers are used.
                Either an int which denotes where MoE layers are used from to the end, or a list
                of ints denoting the specific blocks (both use 0-indexing).
            img_size (int or tuple[int, int]): Input image size.
            patch_size (int or tuple[int, int]): Patch size.
            in_chans (int): Number of image input channels.
            num_classes (int): Number of classes for the classification head.
            global_pool (str): Type of global pooling for the final sequence (default: 'token').
            embed_dim (int): Transformer embedding dimension.
            depth (int): Depth of the transformer.
            num_heads (int): Number of attention heads.
            mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
            qkv_bias (bool): Enable bias for qkv projections if True.
            qk_norm (bool): Enable normalization of query and key in self-attention.
            init_values (float or None): Layer-scale init values (layer-scale enabled if not None).
            class_token (bool): Use a class token.
            no_embed_class (bool): Do not embed class tokens in the patch embedding.
            pre_norm (bool): Apply normalization before self-attention in the transformer block.
            fc_norm (bool or None): Pre-head norm after pool (instead of before).
                If None, enabled when global_pool == 'avg'.
            drop_rate (float): Head dropout rate.
            pos_drop_rate (float): Position embedding dropout rate.
            attn_drop_rate (float): Attention dropout rate.
            drop_path_rate (float): Stochastic depth rate.
            weight_init (str): Weight initialization scheme.
            embed_layer (Callable): Patch embedding layer.
            norm_layer (Callable or None): Normalization layer.
            act_layer (Callable or None): MLP activation layer.
            block_fn (Callable): Transformer block layer.
            mlp_layer (Callable): MLP layer.
        """
        super().__init__()
        assert global_pool in ("", "avg", "token")
        assert class_token or global_pool != "token"
        use_fc_norm = global_pool == "avg" if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class
        self.grad_checkpointing = False

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = (
            nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        )
        embed_len = (
            num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        )
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        # Wrap the mlp_layer in a soft-moe wrapper
        self.num_experts = num_experts
        self.slots_per_expert = slots_per_expert

        moe_mlp_layer = partial(
            SoftMoELayerWrapper,
            layer=mlp_layer,
            dim=embed_dim,
            num_experts=self.num_experts,
            slots_per_expert=self.slots_per_expert,
        )

        # Create a list where each index is the mlp layer class to
        # use at that depth
        self.moe_layer_index = moe_layer_index
        if isinstance(moe_layer_index, list):
            # Only the specified layers in moe_layer_index
            assert len(moe_layer_index) > 0
            assert all([0 <= l < depth for l in moe_layer_index])

            mlp_layers_list = [
                moe_mlp_layer if i in moe_layer_index else mlp_layer
                for i in range(depth)
            ]
        else:
            # All layers including and after moe_layer_index
            assert 0 <= moe_layer_index < depth

            mlp_layers_list = [
                moe_mlp_layer if i >= moe_layer_index else mlp_layer
                for i in range(depth)
            ]

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(
            *[
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    init_values=init_values,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    mlp_layer=mlp_layers_list[i],
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        if weight_init != "skip":
            self.init_weights(weight_init)

    def init_weights(self, mode=""):
        assert mode in ("jax", "jax_nlhb", "moco", "")
        head_bias = -math.log(self.num_classes) if "nlhb" in mode else 0.0
        trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        init_weights_vit_timm(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token"}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r"^cls_token|pos_embed|patch_embed",  # stem and embed
            blocks=[(r"^blocks\.(\d+)", None), (r"^norm", (99999,))],
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes: int, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ("", "avg", "token")
            self.global_pool = global_pool
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def _pos_embed(self, x):
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + self.pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.pos_embed
        return self.pos_drop(x)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = (
                x[:, self.num_prefix_tokens :].mean(dim=1)
                if self.global_pool == "avg"
                else x[:, 0]
            )
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

class Soft_GRPAH_MoEVisionTransformer(nn.Module):
    """Vision Transformer with Soft Mixture of Experts MLP layers.

    From the paper "From Sparse to Soft Mixtures of Experts"
    https://arxiv.org/pdf/2308.00951.pdf

    Code modified from:
    https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
    """

    def __init__(
        self,
        num_experts: int = 128,
        slots_per_expert: int = 1,
        moe_layer_index: int | list[int] = 6,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: str = "token",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        init_values: float | None = None,
        class_token: bool = True,
        no_embed_class: bool = False,
        pre_norm: bool = False,
        fc_norm: bool | None = None,
        drop_rate: float = 0.0,
        pos_drop_rate: float = 0.0,
        patch_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        weight_init: str = "",
        embed_layer: Callable = PatchEmbed,
        norm_layer: Callable | None = None,
        act_layer: Callable | None = None,
        block_fn: Callable = Block,
        mlp_layer: Callable = Mlp,
        pretrained_cfg=None,
        pretrained_cfg_overlay=None,
        alpha=0.9,
        beta=0.9,
        t=0.5,
    ):
        """
        Args:
            num_experts (int): Number of experts in MoE layers.
            slots_per_expert (int): Number of token slots per expert.
            moe_layer_index (int or list[int]): Block depth indices where MoE layers are used.
                Either an int which denotes where MoE layers are used from to the end, or a list
                of ints denoting the specific blocks (both use 0-indexing).
            img_size (int or tuple[int, int]): Input image size.
            patch_size (int or tuple[int, int]): Patch size.
            in_chans (int): Number of image input channels.
            num_classes (int): Number of classes for the classification head.
            global_pool (str): Type of global pooling for the final sequence (default: 'token').
            embed_dim (int): Transformer embedding dimension.
            depth (int): Depth of the transformer.
            num_heads (int): Number of attention heads.
            mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
            qkv_bias (bool): Enable bias for qkv projections if True.
            qk_norm (bool): Enable normalization of query and key in self-attention.
            init_values (float or None): Layer-scale init values (layer-scale enabled if not None).
            class_token (bool): Use a class token.
            no_embed_class (bool): Do not embed class tokens in the patch embedding.
            pre_norm (bool): Apply normalization before self-attention in the transformer block.
            fc_norm (bool or None): Pre-head norm after pool (instead of before).
                If None, enabled when global_pool == 'avg'.
            drop_rate (float): Head dropout rate.
            pos_drop_rate (float): Position embedding dropout rate.
            attn_drop_rate (float): Attention dropout rate.
            drop_path_rate (float): Stochastic depth rate.
            weight_init (str): Weight initialization scheme.
            embed_layer (Callable): Patch embedding layer.
            norm_layer (Callable or None): Normalization layer.
            act_layer (Callable or None): MLP activation layer.
            block_fn (Callable): Transformer block layer.
            mlp_layer (Callable): MLP layer.
        """
        super().__init__()
        print( f">>>>>>>> init sftmoe graph vit with alpha {alpha}, beta {beta}, t {t} <<<<<<<<")
        assert global_pool in ("", "avg", "token")
        assert class_token or global_pool != "token"
        use_fc_norm = global_pool == "avg" if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class
        self.grad_checkpointing = False

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = (
            nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        )
        embed_len = (
            num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        )
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        # Wrap the mlp_layer in a soft-moe wrapper
        self.num_experts = num_experts
        self.slots_per_expert = slots_per_expert

        if t > 0: 
            moe_mlp_layer = partial(
                Soft_GRAPH_threshold_MoELayerWrapper,
                layer=mlp_layer,
                dim=embed_dim,
                num_experts=self.num_experts,
                slots_per_expert=self.slots_per_expert,
                alpha=alpha,
                beta=beta,
                t=t,
            )
        else:
            moe_mlp_layer = partial(
                Soft_GRAPH_MoELayerWrapper,
                layer=mlp_layer,
                dim=embed_dim,
                num_experts=self.num_experts,
                slots_per_expert=self.slots_per_expert,
                alpha=alpha,
                beta=beta,
            )

        # Create a list where each index is the mlp layer class to
        # use at that depth
        self.moe_layer_index = moe_layer_index
        if isinstance(moe_layer_index, list):
            # Only the specified layers in moe_layer_index
            assert len(moe_layer_index) > 0
            assert all([0 <= l < depth for l in moe_layer_index])

            mlp_layers_list = [
                moe_mlp_layer if i in moe_layer_index else mlp_layer
                for i in range(depth)
            ]
        else:
            # All layers including and after moe_layer_index
            assert 0 <= moe_layer_index < depth

            mlp_layers_list = [
                moe_mlp_layer if i >= moe_layer_index else mlp_layer
                for i in range(depth)
            ]

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(
            *[
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    init_values=init_values,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    mlp_layer=mlp_layers_list[i],
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        if weight_init != "skip":
            self.init_weights(weight_init)

    def init_weights(self, mode=""):
        assert mode in ("jax", "jax_nlhb", "moco", "")
        head_bias = -math.log(self.num_classes) if "nlhb" in mode else 0.0
        trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        init_weights_vit_timm(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token"}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r"^cls_token|pos_embed|patch_embed",  # stem and embed
            blocks=[(r"^blocks\.(\d+)", None), (r"^norm", (99999,))],
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes: int, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ("", "avg", "token")
            self.global_pool = global_pool
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def _pos_embed(self, x):
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + self.pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.pos_embed
        return self.pos_drop(x)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = (
                x[:, self.num_prefix_tokens :].mean(dim=1)
                if self.global_pool == "avg"
                else x[:, 0]
            )
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x