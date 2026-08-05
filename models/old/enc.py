##IMPORT 
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
from functools import partial
#from transformers import ViTImageProcessor, ViTForImageClassification

class LayerNormGeneral(nn.Module):

    def __init__(self, affine_shape=None, normalized_dim=(-1, ), scale=True, 
        bias=True, eps=1e-5):
        super().__init__()
        self.normalized_dim = normalized_dim
        self.use_scale = scale
        self.use_bias = bias
        self.weight   = nn.Parameter(torch.ones(affine_shape)) if scale else None
        self.bias     = nn.Parameter(torch.zeros(affine_shape)) if bias else None
        self.eps      = eps

    def forward(self, x):
        c = x - x.mean(self.normalized_dim, keepdim=True)
        s = c.pow(2).mean(self.normalized_dim, keepdim=True)
        x = c / torch.sqrt(s + self.eps)
        if self.use_scale:
            x = x * self.weight
        if self.use_bias:
            x = x + self.bias
        return x


class Mlp(nn.Module):
    """ MLP as used in MetaFormer models, eg Transformer, MLP-Mixer, PoolFormer, MetaFormer baslines and related networks.
    Mostly copied from timm.
    """
    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=nn.GELU, drop=0., bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = (drop,drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.norm = nn.LayerNorm(dim)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.norm(x)
        x = self.drop2(x)
        return x.permute(0, 3, 1, 2)

class Scale(nn.Module):

    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale

class Attention(nn.Module):
    """
    Transformer: https://arxiv.org/abs/1706.03762.
    """
    def __init__(self, dim, head_dim=32, num_heads=None, qkv_bias=False,
        attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim ** -0.5   #embed_dim/head_number = head

        self.num_heads = num_heads if num_heads else dim // head_dim

        if self.num_heads == 0:
            self.num_heads = 1
        
        self.attention_dim = self.num_heads * self.head_dim

        self.qkv        = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop  = nn.Dropout(attn_drop)
        self.proj       = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop  = nn.Dropout(proj_drop)

        
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        N = H * W
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.attention_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x.permute(0, 3, 1, 2)


class ConvBlock(nn.Module):

    def __init__(self, dim, drop=0.):
        super().__init__()

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding='same',groups=dim) # depthwise conv

        #self.norm = nn.BatchNorm2d(dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop) if drop > 0. else nn.Identity()

    def forward(self, x):
        
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.act(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop_path(x)
        x = x.permute(0, 3, 1, 2)
        return x

    
class SepConv(nn.Module):

    def __init__(self, dim, drop=0.):
        super().__init__()

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding='same',groups=dim) # depthwise conv

        #self.norm = nn.BatchNorm2d(dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop) if drop > 0. else nn.Identity()

    def forward(self, x):
        
        x = self.dwconv(x)#self.dwconv2(x)+self.dwconv3(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.act(x)

        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop_path(x)
        x = x.permute(0, 3, 1, 2)
        return x

class Downsampling(nn.Module):

    def __init__(self, in_channels, out_channels, 
        kernel_size, stride=1, padding=0, 
        pre_norm=None, pre_permute=False):
        super().__init__()

        self.pre_norm = pre_norm(in_channels) if pre_norm else nn.Identity()
        self.pre_permute = pre_permute
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.norm = nn.LayerNorm(out_channels)
        self.down = nn.MaxPool2d(2,2)
        self.act= nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.act(x)
        x = self.down(x)
        return x
        

DOWNSAMPLE_LAYERS_FOUR_STAGES = [partial(Downsampling,
            kernel_size=3, stride=2, padding=1,
            )] + \
            [partial(Downsampling,
                kernel_size=3, stride=2, padding=1, 
                pre_norm=partial(LayerNormGeneral, bias=False, eps=1e-6), pre_permute=True
            )]*3

class EncoderBlock(nn.Module):
    """Single encoder block with token mixer + conv block + residual connection."""

    def __init__(self, dim, token_mixer=nn.Identity, cblock=ConvBlock, drop=0.0, drop_path=0.0):
        super().__init__()
        self.token_mixer = token_mixer(dim=dim, drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.cblock = cblock(dim=dim, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.token_mixer(x))
        x = x + self.drop_path2(self.cblock(x))
        return x


class Encoder(nn.Module):
    """Encoder with 4 stages, each containing a single EncoderBlock."""

    def __init__(self,
                 in_chans=3,
                 dims=[64, 128, 256, 512],
                 downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
                 token_mixers=None,
                 drop_path_rate=0.0):
        super().__init__()
        assert token_mixers is not None, "You must provide a list of token_mixers for 4 stages"
        assert len(token_mixers) == 4, "Expected 4 token mixers (one per stage)"

        self.num_stages = 4
        down_dims = [in_chans] + dims

        # Build downsampling layers
        self.downsample_layers = nn.ModuleList([
            downsample_layers[i](down_dims[i], down_dims[i + 1]) for i in range(self.num_stages)
        ])

        # Drop path rate schedule (just 4 values for 4 blocks)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_stages)]

        # Each stage = one EncoderBlock
        self.stages = nn.ModuleList([EncoderBlock(dim=dims[i], token_mixer=token_mixers[i], drop_path=dp_rates[i]) for i in range(self.num_stages)])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> list:
        """Forward pass through encoder. Returns list of feature maps from each stage."""
        features = []
        for i in range(self.num_stages):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            features.append(x)
        return features


def encoder_function():
    """Factory method to create the encoder with fixed 4 stages."""
    return Encoder(
        dims=[64, 128, 256, 512],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
    )


if __name__ == "__main__":
    model = encoder_function()
    x = torch.rand(2, 3, 256, 256)
    features = model(x)
    print("Feature shapes per stage:")
    for i, feat in enumerate(features):
        print(f" Stage {i+1}: {tuple(feat.shape)}")
