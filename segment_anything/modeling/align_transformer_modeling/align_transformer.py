import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from .positional_encoding import PositionalEncoding
from .transformer_layers import *
import torchvision.models as models
from torchvision.models.resnet import Bottleneck
    

class AlignMLP(nn.Module):
    def __init__(self, num_layers:int, dim_in: int, dim_ff: int, activation=F.relu):
        super().__init__()
        self.layers = nn.ModuleList([FFN(dim_in, dim_ff, activation) for _ in range(num_layers)])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : shape [B, D, H, W]
        bs, dim, h, w = x.shape
        x_flat = x.flatten(2).transpose(1, 2).contiguous()  # [B, H*W, D]
        for layer in self.layers:
            x_flat = layer(x_flat)
        x = x_flat.transpose(1, 2).contiguous().view(bs, dim, h, w)    # [B, D, H, W]
        return x
    

class AlignCNN(nn.Module):
    def __init__(self, num_layers:int, dim:int):
        super().__init__()
        self.layers = nn.ModuleList([CAResBlock(dim, dim) for _ in range(num_layers)])
        self.norm = LayerNorm2d(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : shape [B, D, H, W]
        for layer in self.layers:
            x = layer(x)
            x = self.norm(x)
        return x


class Align_New(nn.Module):
    def __init__(self, C: int, dim: int, num_align_layers: int = 3, pretrained_backbone: bool = False):
        super().__init__()
        assert num_align_layers >= 1, "num_align_layers must be >= 1"


        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        backbone = models.resnet50(weights=weights)

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool  # 1024->512
        )
        self.layer1 = backbone.layer1      # -> 512×512
        self.layer2 = backbone.layer2      # -> 256×256
        self.layer3 = backbone.layer3


        self.backbone_out_channels = 1024



        blocks = [CAResBlock(self.backbone_out_channels + dim, dim)]
        for _ in range(num_align_layers - 1):
            blocks.append(CAResBlock(dim, dim))
        self.align_layers = nn.ModuleList(blocks)

        self.norm = LayerNorm2d(dim)
        self.dim = dim
        self.C = C

    @torch.no_grad()
    def _check_inputs(self, img: torch.Tensor, feat: torch.Tensor):
        assert img.ndim == 4 and img.shape[1] == 3, f"expected img shape [N,3,H,W], got {tuple(img.shape)}"
        assert img.shape[2:] == (1024, 1024), f"expected img spatial size 1024x1024, got {img.shape[2:]}"
        assert feat.ndim == 4 and feat.shape[1:] == (self.dim, 64, 64), \
            f"expected feat shape [N,{self.dim},64,64], got {tuple(feat.shape)}"
        assert img.shape[0] == feat.shape[0], "img and feat must have the same batch size"

    def forward(self, img: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img:  [N, 3, 1024, 1024]
            feat: [N, dim, 64, 64]
        Returns:
            x:    [N, dim, 64, 64]
        """
        self._check_inputs(img, feat)


        x = self.stem(img)       # [N, 64, 512, 512]
        x = self.layer1(x)       # [N, 256, 512, 512]
        x = self.layer2(x)       # [N, 512, 256, 256]
        x = self.layer3(x)       # [N, 1024, 64, 64]


        x = torch.cat([x, feat], dim=1)  # [N, 1024+dim, 64, 64]


        for blk in self.align_layers:
            x = blk(x)
            x = self.norm(x)

        return x


class CNNStage1(nn.Module):
    def __init__(self, num_repeats: int = 3, use_bn: bool = True):
        super().__init__()

        assert num_repeats >= 1, "num_repeats must be at least 1"

        blocks = []

        blocks.append(Bottleneck(inplanes=256, planes=64))

        for _ in range(num_repeats - 1):
            blocks.append(Bottleneck(inplanes=256, planes=64))

        self.stage = nn.Sequential(*blocks)

        self.norm = nn.BatchNorm2d(256) if use_bn else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.stage(x)   # -> [B, 256, H, W]
        x = self.norm(x)    # -> [B, 256, H, W]
        return x

class AlignTransformerBlock(nn.Module):
    def __init__(self, 
                 embed_dim, 
                 num_heads: int = 8,
                 num_queries: int = 16, 
                 ff_dim: int = 1024):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.ff_dim = ff_dim

        self.read_from_pixel = CrossAttention(self.embed_dim,
                                              self.num_heads,
                                              add_pe_to_qkv=[True, True, False])
        self.self_attn = SelfAttention(self.embed_dim,
                                       self.num_heads,
                                       add_pe_to_qkv=[True, True, False])
        self.ffn = FFN(self.embed_dim, self.ff_dim)
        self.read_from_query = CrossAttention(self.embed_dim,
                                              self.num_heads,
                                              add_pe_to_qkv=[True, True, False],
                                              norm=False)
        self.pixel_ffn = PixelFFN(self.embed_dim)

    def forward(
            self,
            query: torch.Tensor,
            pixel: torch.Tensor,
            query_pe: torch.Tensor,
            pixel_pe: torch.Tensor,) -> (torch.Tensor, torch.Tensor): # type: ignore
        # query: [B, num_queries, embed_dim]
        # pixel: [B, embed_dim, H, W]
        # query_pe: [B, num_queries, embed_dim]
        # pixel_pe: [B, H*W, embed_dim]
        
        r = pixel
        pixel_flat = pixel.flatten(2, 3).transpose(1, 2).contiguous()  # [B, H*W, embed_dim]
        query, _ = self.read_from_pixel(query,
                                        pixel_flat,
                                        query_pe,
                                        pixel_pe,
                                        need_weights=False)
        query = self.self_attn(query, query_pe)
        query = self.ffn(query)

        pixel_flat, _ = self.read_from_query(pixel_flat,
                                             query,
                                             pixel_pe,
                                             query_pe,
                                             need_weights=False)
        pixel = self.pixel_ffn(pixel, pixel_flat)
        pixel = pixel + r

        return query, pixel


class AlignTransformer(nn.Module):
    '''
    input tensors (pixel): [B, input_dim, H, W]
    output tensors (pixel): [B, input_dim, H, W]
    trainable embeddings (distribution query) for cross-attention: [B, num_queries, embed_dim]
    
    pipeline:
    1. n transformer blocks:
        1.1. distribution query + pe = Q, pixel + pe = K, pixel = V. --> cross-attention + FFN --> updated distribution query
        1.2. pixel + pe = Q, updated distribution query + pe = K, updated distribution query = V. --> cross-attention + pixel FFN --> updated pixel
    2. output n-updated pixel as the output [B, embed_dim, H, W]
    '''
    def __init__(self, 
                 num_blocks: int = 3,
                 input_dim: int = 256,
                 embed_dim: int = 512, 
                 num_heads: int = 8,
                 num_queries: int = 16,
                 pixel_pe_scale: float = 32,
                 pixel_pe_temperature: float = 128,):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = input_dim
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads  # num of attention heads
        self.num_queries = num_queries  # num of distribution queries
        self.query_grid = self._infer_query_grid(self.num_queries)
        
        self.input_norm = LayerNorm2d(self.input_dim)

        # query initialization and embedding
        self.query_init = nn.Embedding(self.num_queries, self.embed_dim)                # [num_queries, embed_dim]
        self.query_emb = nn.Parameter(torch.zeros(self.num_queries, self.embed_dim))    # [num_queries, embed_dim]

        # projection from object summaries to query initialization and embedding
        self.summary_to_distribution_init = nn.Linear(self.input_dim, self.embed_dim)
        self.summary_to_distribution_emb = nn.Linear(self.input_dim, self.embed_dim)

        self.pixel_pe_scale = pixel_pe_scale  # scaling factor for pixel positional embedding
        self.pixel_pe_temperature = pixel_pe_temperature
        
        self.pixel_init_proj = nn.Conv2d(self.input_dim, self.embed_dim, kernel_size=3, padding=1)
        self.pixel_emb_proj = nn.Conv2d(self.input_dim, self.embed_dim, kernel_size=3, padding=1)
        
        self.spatial_pe = PositionalEncoding(self.embed_dim,
                                             scale=self.pixel_pe_scale,
                                             temperature=self.pixel_pe_temperature,
                                             channel_last=False,
                                             transpose_output=False)

        # transformer blocks
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList(
            AlignTransformerBlock(embed_dim=self.embed_dim,
                                  num_heads=self.num_heads,
                                  num_queries=self.num_queries,
                                  ff_dim=self.embed_dim * 4) for _ in range(self.num_blocks))
        
        self.neck = nn.Sequential(
            LayerNorm2d(self.embed_dim),
            nn.Conv2d(
                self.embed_dim,
                self.output_dim,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(self.output_dim),
            nn.Conv2d(
                self.output_dim,
                self.output_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
        )
        self.output_scale = nn.Parameter(torch.full((self.output_dim,), 1e-3))
        
        # self.mask_pred = nn.ModuleList(
        #     nn.Sequential(nn.ReLU(), nn.Conv2d(self.embed_dim, 1, kernel_size=1))
        #     for _ in range(self.num_blocks + 1))
    
    def _infer_query_grid(self, num_queries: int) -> tuple[int, int]:
        side = int(math.sqrt(num_queries))
        for h in range(side, 0, -1):
            if num_queries % h == 0:
                return h, num_queries // h
        return 1, num_queries

    def _pooling(self, x: torch.Tensor, k: int) -> torch.Tensor:
        # x: [B, C, H, W]
        # output: [B, k, C]
        # k : num_queries
        gh, gw = self.query_grid
        x = F.adaptive_avg_pool2d(x, output_size=(gh, gw))  # [B, C, gh, gw]
        x = x.flatten(2).transpose(1, 2).contiguous()       # [B, gh*gw, C]
        if x.shape[1] != k:
            x = x.transpose(1, 2).contiguous()              # [B, C, gh*gw]
            x = F.adaptive_avg_pool1d(x, k)                 # [B, C, k]
            x = x.transpose(1, 2).contiguous()              # [B, k, C]
        return x
        
    def forward(self,
                pixel: torch.Tensor) -> torch.Tensor:
        # pixel: B, input_dim, H, W
        
        bs, D, H, W = pixel.shape
        assert D == self.input_dim, f"expect pixel with {self.input_dim} channels, but got {D}"
        
        r = pixel
        pixel = self.input_norm(pixel)

        # positional embeddings for distribution queries
        pixel_flat_pool = self._pooling(pixel, self.num_queries) # B, num_queries, input_dim
        distribution_init = self.summary_to_distribution_init(pixel_flat_pool)                 # B, num_queries, embed_dim
        distribution_emb = self.summary_to_distribution_emb(pixel_flat_pool)                   # B, num_queries, embed_dim
        query = self.query_init.weight.unsqueeze(0).expand(bs, -1, -1) + distribution_init      # B, num_queries, embed_dim
        query_emb = self.query_emb.unsqueeze(0).expand(bs, -1, -1) + distribution_emb    # B, num_queries, embed_dim

        # positional embeddings for pixel features
        pixel_init = self.pixel_init_proj(pixel)    # B, embed_dim, H, W
        pixel_emb = self.pixel_emb_proj(pixel)      # B, embed_dim, H, W
        pixel_pe = self.spatial_pe(pixel)           # B, embed_dim, H, W
        pixel_pe = pixel_pe + pixel_emb
        pixel_pe = pixel_pe.flatten(2, 3).transpose(1, 2).contiguous()  # B, H*W, embed_dim

        pixel = pixel_init


        for i in range(self.num_blocks):
            query, pixel  =  self.blocks[i](query,
                                            pixel,
                                            query_emb,
                                            pixel_pe,)
            
        pixel = self.neck(pixel)    # [B, output_dim, H, W]
        pixel = r + self.output_scale.view(1, -1, 1, 1) * pixel
        return pixel
    

class AlignTransformerBlockPlus(nn.Module):
    def __init__(self, 
                 embed_dim, 
                 num_heads: int = 8,):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.read_from_feature = CrossAttention(self.embed_dim,
                                              self.num_heads,
                                              add_pe_to_qkv=[True, True, False])
        self.pixel_self_attn = SelfAttention(self.embed_dim,
                                             self.num_heads,
                                             add_pe_to_qkv=[True, True, False])
        self.pixel_ffn = PixelFFN(self.embed_dim)
        self.pixel_norm = LayerNorm2d(self.embed_dim)
        self.read_from_pixel = CrossAttention(self.embed_dim,
                                              self.num_heads,
                                              add_pe_to_qkv=[True, True, False],
                                              norm=False)
        self.feature_self_attn = SelfAttention(self.embed_dim,
                                               self.num_heads,
                                               add_pe_to_qkv=[True, True, False])
        self.feature_ffn = PixelFFN(self.embed_dim)
        self.feature_norm = LayerNorm2d(self.embed_dim)

    def forward(
            self,
            pixel: torch.Tensor,
            feature: torch.Tensor,
            pixel_pe: torch.Tensor,
            feature_pe: torch.Tensor) -> (torch.Tensor, torch.Tensor): # type: ignore
        # pixel: [B, embed_dim, H, W]
        # feature: [B, embed_dim, H, W]
        # feature_pe: [B, H*W, embed_dim]
        # pixel_pe: [B, H*W, embed_dim]
        
        r = feature
        pixel_flat = pixel.flatten(2, 3).transpose(1, 2).contiguous()  # [B, H*W, embed_dim]
        feature_flat = feature.flatten(2, 3).transpose(1, 2).contiguous()  # [B, H*W, embed_dim]
        
        pixel_flat, _ = self.read_from_feature(pixel_flat,
                                          feature_flat,
                                          pixel_pe,
                                          feature_pe,
                                          need_weights=False)
        pixel_flat = self.pixel_self_attn(pixel_flat, pixel_pe)
        pixel = self.pixel_ffn(pixel, pixel_flat)       # [B, embed_dim, H, W]
        pixel = self.pixel_norm(pixel)
        
        pixel_flat = pixel.flatten(2, 3).transpose(1, 2).contiguous()  # [B, H*W, embed_dim]

        feature_flat, _ = self.read_from_pixel(feature_flat,
                                             pixel_flat,
                                             feature_pe,
                                             pixel_pe,
                                             need_weights=False)
        feature_flat = self.feature_self_attn(feature_flat, feature_pe)
        feature = self.feature_ffn(feature, feature_flat)
        feature = self.feature_norm(feature)
        feature = feature + r
        
        bs, d, h, w = pixel.shape
        pixel = pixel_flat.view(bs, h, w, d)
        pixel = pixel.permute(0, 3, 1, 2).contiguous()    # [B, embed_dim, H, W]

        return pixel, feature
    
class AlignTransformerPlus(nn.Module):
    '''
    input tensors (pixel): [B, 3, H', W']
    input tensors (feature): [B, input_dim, H, W]
    output tensors (feature): [B, input_dim, H, W]
    
    pipeline:
    1. pixel -> Convs. -> [B, embed_dim, H, W]
    2. n transformer blocks:
        1.1. pixel + pe = Q, feature + pe = K, feature = V. --> cross-attention + FFN --> updated pixel
        1.2. feature + pe = Q, updated pixel + pe = K, updated pixel = V. --> cross-attention + FFN --> updated pixel
    3. output n-updated feature as the output [B, embed_dim, H, W]
    '''
    def __init__(self, 
                 num_blocks: int = 5,
                 patch_size: int = 16,
                 input_feature_dim: int = 256,
                 embed_dim: int = 512, 
                 num_heads: int = 8,
                 pe_scale: float = 32,
                 pe_temperature: float = 128,):
        super().__init__()

        self.input_dim = input_feature_dim
        self.output_dim = input_feature_dim
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads  # num of attention heads
        
        self.input_norm = LayerNorm2d(self.input_dim)

        # Init pixel
        self.init_norm_feature = LayerNorm2d(self.embed_dim)
        self.init_norm_pixel = LayerNorm2d(self.embed_dim)

        self.pe_scale = pe_scale  # scaling factor for pixel positional embedding
        self.pe_temperature = pe_temperature
        
        self.feature_init_proj = nn.Conv2d(self.input_dim, self.embed_dim, kernel_size=3, padding=1)
        self.feature_emb_proj = nn.Conv2d(self.input_dim, self.embed_dim, kernel_size=3, padding=1)
        
        self.spatial_pe = PositionalEncoding(self.embed_dim,
                                             scale=self.pe_scale,
                                             temperature=self.pe_temperature,
                                             channel_last=False,
                                             transpose_output=False)
        
        self.patch_size = patch_size
        self.pixel_conv_0 = nn.Conv2d(3, self.embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pixel_conv_1 = nn.Sequential(
            LayerNorm2d(self.embed_dim),
            nn.ReLU(),
            nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=3, padding=1),
        )
        self.pixel_conv_2 = nn.Sequential(
            LayerNorm2d(self.embed_dim),
            nn.ReLU(),
            nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=1),
        )
        
        self.pixel_init_proj = nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=3, padding=1)
        self.pixel_emb_proj = nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=3, padding=1)

        # transformer blocks
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList(
            AlignTransformerBlockPlus(embed_dim=self.embed_dim,
                                  num_heads=self.num_heads) for _ in range(self.num_blocks))
        
        self.neck = nn.Sequential(
            LayerNorm2d(self.embed_dim),
            nn.Conv2d(
                self.embed_dim,
                self.output_dim,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(self.output_dim),
            nn.Conv2d(
                self.output_dim,
                self.output_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
        )
        
        
    def forward(self,
                pixel: torch.Tensor,
                feature: torch.Tensor) -> (torch.Tensor, Dict[str, torch.Tensor]): # type: ignore
        # pixel: B, 3, H', W'
        # feature: B, input_dim, H, W
        
        bs, D, H, W = feature.shape
        assert D == self.input_dim, f"expect pixel with {self.embed_dim} channels, but got {D}"
        
        r_globel = feature
        feature = self.input_norm(feature)
        
        pixel = self.pixel_conv_0(pixel)    # [B, embed_dim, H, W]
        assert pixel.shape[-2:] == feature.shape[-2:], f"expect pixel with {feature.shape[-2:]} resolution, but got {pixel.shape[-2:]}"
        
        pixel = self.pixel_conv_1(pixel) + pixel
        pixel = self.pixel_conv_2(pixel) + pixel    # [B, embed_dim, H, W]

        # positional embeddings for pixel features
        pixel_init = self.pixel_init_proj(pixel)    # B, embed_dim, H, W
        pixel_emb = self.pixel_emb_proj(pixel)      # B, embed_dim, H, W
        feature_init = self.feature_init_proj(feature)    # B, embed_dim, H, W
        feature_emb = self.feature_emb_proj(feature)      # B, embed_dim, H, W
        
        pixel_pe = self.spatial_pe(pixel)           # B, embed_dim, H, W
        pixel_pe = pixel_pe + pixel_emb
        pixel_pe = pixel_pe.flatten(2, 3).transpose(1, 2).contiguous()  # B, H*W, embed_dim
        pixel = pixel_init
        pixel = self.init_norm_pixel(pixel)

        feature_pe = self.spatial_pe(feature)           # B, embed_dim, H, W
        feature_pe = feature_pe + feature_emb
        feature_pe = feature_pe.flatten(2, 3).transpose(1, 2).contiguous()  # B, H*W, embed_dim
        feature = feature_init
        feature = self.init_norm_feature(feature)

        for i in range(self.num_blocks):
            pixel, feature  =  self.blocks[i](pixel,
                                            feature,
                                            pixel_pe,
                                            feature_pe,)
            
        feature = self.neck(feature)    # [B, output_dim, H, W]
        feature = feature + r_globel
        return feature

if __name__ == "__main__":
    pixel = torch.rand(2, 3, 256, 256).cuda()
    feature = torch.rand(2, 256, 16, 16).cuda()
    model = AlignTransformerPlus().cuda()
    out = model(pixel, feature)
    print(out.shape)
