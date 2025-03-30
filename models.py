# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial


from timm.models.vision_transformer import _cfg
from softmax import VisionTransformer
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
# from xcit import XCiT, HDPXCiT

class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)


        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x, x_dist
        else:
            # during inference, return the average of both classifier predictions
            return (x + x_dist) / 2

# register model with timms to be able to call it from "create_model" using its function name
# but mainly edit the model from softmax.py
@register_model
def deit_tiny_patch16_224(pretrained=True, **kwargs):
    from softmax import VisionTransformer
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs) # Tan's NOTE: in the original code, num_heads = 3 here
    model.default_cfg = _cfg()
    return model

@register_model
def soft_moe_vit_tiny(pretrained=False,
    num_experts=128, slots_per_expert=1, moe_layer_index=6, **kwargs):
    from softmax import SoftMoEVisionTransformer
    model = SoftMoEVisionTransformer(
        num_experts=num_experts,
        slots_per_expert=slots_per_expert,
        moe_layer_index=moe_layer_index,
        embed_dim=192,
        depth=12,
        num_heads=3,
        **kwargs,
    )
    return model

@register_model
def soft_graph_moe_vit_tiny(pretrained=False,
    num_experts=128, slots_per_expert=1, moe_layer_index=6, **kwargs):
    from softmax import Soft_GRPAH_MoEVisionTransformer
    model = Soft_GRPAH_MoEVisionTransformer(
        num_experts=num_experts,
        slots_per_expert=slots_per_expert,
        moe_layer_index=moe_layer_index,
        embed_dim=192,
        depth=12,
        num_heads=3,
        **kwargs,
    )
    return model

# def soft_moe_vit_small(
#     num_experts=128, slots_per_expert=1, moe_layer_index=6, **kwargs
# ) -> SoftMoEVisionTransformer:
#     return SoftMoEVisionTransformer(
#         num_experts=num_experts,
#         slots_per_expert=slots_per_expert,
#         moe_layer_index=moe_layer_index,
#         embed_dim=384,
#         depth=12,
#         num_heads=6,
#         **kwargs,
#     )


# def soft_moe_vit_base(
#     num_experts=128, slots_per_expert=1, moe_layer_index=6, **kwargs
# ) -> SoftMoEVisionTransformer:
#     return SoftMoEVisionTransformer(
#         num_experts=num_experts,
#         slots_per_expert=slots_per_expert,
#         moe_layer_index=moe_layer_index,
#         embed_dim=768,
#         depth=12,
#         num_heads=12,
#         **kwargs,
#     )


# def soft_moe_vit_large(
#     num_experts=128, slots_per_expert=1, moe_layer_index=12, **kwargs
# ) -> SoftMoEVisionTransformer:
#     return SoftMoEVisionTransformer(
#         num_experts=num_experts,
#         slots_per_expert=slots_per_expert,
#         moe_layer_index=moe_layer_index,
#         embed_dim=1024,
#         depth=24,
#         num_heads=16,
#         **kwargs,
#     )


# def soft_moe_vit_huge(
#     num_experts=128, slots_per_expert=1, moe_layer_index=16, **kwargs
# ) -> SoftMoEVisionTransformer:
#     return SoftMoEVisionTransformer(
#         num_experts=num_experts,
#         slots_per_expert=slots_per_expert,
#         moe_layer_index=moe_layer_index,
#         embed_dim=1280,
#         depth=32,
#         num_heads=16,
#         **kwargs,
#     )
