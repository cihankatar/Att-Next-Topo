##IMPORT 
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from functools import partial

class Scale(nn.Module):

    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale
        

class LayerNormGeneral(nn.Module):

    def __init__(self, affine_shape=None, normalized_dim=(-1, ), scale=True, 
        bias=True, eps=1e-5):
        super().__init__()
        self.normalized_dim = normalized_dim
        self.use_scale = scale
        self.use_bias = bias
        self.weight = nn.Parameter(torch.ones(affine_shape)) if scale else None
        self.bias = nn.Parameter(torch.zeros(affine_shape)) if bias else None
        self.eps = eps

    def forward(self, x):
        c = x - x.mean(self.normalized_dim, keepdim=True)
        s = c.pow(2).mean(self.normalized_dim, keepdim=True)
        x = c / torch.sqrt(s + self.eps)
        if self.use_scale:
            x = x * self.weight
        if self.use_bias:
            x = x + self.bias
        return x



class LayerNormWithoutBias(nn.Module):

    def __init__(self, normalized_shape, eps=1e-5, **kwargs):
        super().__init__()
        self.eps = eps
        self.bias = None
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape
    def forward(self, x):
        return F.layer_norm(x, self.normalized_shape, weight=self.weight, bias=self.bias, eps=self.eps)


class Mlp(nn.Module):
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

class Attention(nn.Module):
    """
    Transformer: https://arxiv.org/abs/1706.03762.
    """
    def __init__(self, dim, head_dim=32, num_heads=None, qkv_bias=False,
        attn_drop=0.2, proj_drop=0.2, proj_bias=False, **kwargs):
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

class SepConv(nn.Module):
    def __init__(self, dim, drop=0.):
        super().__init__()

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding='same',groups=dim) # depthwise conv
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

class upsampling(nn.Module):

    def __init__(self, in_channels, out_channels, 
        kernel_size, stride=1, padding=0, 
        pre_norm=None, post_norm=None, pre_permute=False):
        super().__init__()
        self.pre_norm = pre_norm(in_channels) if pre_norm else nn.Identity()
        self.pre_permute = pre_permute
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.LayerNorm(out_channels)
        self.act = nn.GELU()
        self.up   = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.post_norm = post_norm(out_channels) if post_norm else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x= self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.act(x)
        x= self.up(x)
        return x

UPSAMPLE_LAYERS_FOUR_STAGES =[partial(upsampling,
                kernel_size=3, padding='same', 
                pre_norm=partial(LayerNormGeneral, bias=False, eps=1e-6)
            )]*3

class DecoderBlock(nn.Module):
    """
    A single decoder block with:
    - Token mixer (e.g. Attention or Identity)
    - DropPath regularization
    - Convolutional block (for local feature refinement)
    """
    def __init__(self,
                 dim,
                 token_mixer=nn.Identity,
                 conv_block=ConvBlock,
                 drop=0.0,
                 drop_path=0.0,
                 layer_scale_init_value=None,
                 res_scale_init_value=None):
        super().__init__()

        # --- Token mixer (global or local feature interaction) ---
        self.token_mixer = token_mixer(dim=dim, drop=drop)

        # --- DropPath regularization (stochastic depth) ---
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # --- Local convolutional block ---
        self.conv_block = conv_block(dim=dim, drop=0)

    def forward(self, x):
        # Residual connection around token mixer
        x = x + self.drop_path1(self.token_mixer(x))

        # Residual connection around convolution block
        x = x + self.drop_path2(self.conv_block(x))

        return x

    
class Decoder(nn.Module):
    def __init__(self,
                 num_classes=1000, 
                 dims=[512, 256, 128, 64],
                 up_layers=UPSAMPLE_LAYERS_FOUR_STAGES,
                 token_mixers=nn.Identity,
                 drop_path_rate=0.0,
                 head_dropout=0.0, 
                 layer_scale_init_values=None,
                 res_scale_init_values=[None, None, 1.0, 1.0],
                 head_fn=nn.Linear,
                 **kwargs):
        super().__init__()

        # --- Force depth=1 for all stages ---
        self.dims = dims
        self.depths = [1] * len(dims)
        self.num_stage = len(self.depths)

        # --- Upsampling layers ---
        self.up_layers = nn.ModuleList(
            [up_layers[i](dims[i], dims[i + 1]) for i in range(self.num_stage - 1)]
        )
        self.up_layers.append(up_layers[-1](dims[-1], dims[-1]))  # final stage

        # --- DropPath schedule ---
        dp_rates = torch.linspace(0, drop_path_rate, sum(self.depths)).tolist()

        # --- Build decoder stages ---
        self.stages = nn.ModuleList()
        cur = 0
        for i, dim in enumerate(self.dims):
            stage_blocks = [
                DecoderBlock(
                    dim=dim,
                    token_mixer=token_mixers[i],
                    drop_path=dp_rates[cur + j],
                    layer_scale_init_value=layer_scale_init_values[i] if layer_scale_init_values else None,
                    res_scale_init_value=res_scale_init_values[i]
                )
                for j in range(self.depths[i])
            ]
            self.stages.append(nn.Sequential(*stage_blocks))
            cur += self.depths[i]

        # --- Prediction head ---
        self.head = head_fn(dims[-1], num_classes, head_dropout=head_dropout) \
                    if head_dropout > 0.0 else head_fn(dims[-1], num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"norm"}

    def forward(self, x, skip_connections):
        for i in range(self.num_stage):
            x = self.stages[i](x)
            x = self.up_layers[i](x)
            if i < self.num_stage - 1:
                x = skip_connections[i] + x
        return x

def decoder_function(pretrained=False, **kwargs):
    model = Decoder(
        dims=[512, 256, 128, 64],
        token_mixers=[Attention, Attention, SepConv, SepConv],
        **kwargs
    )
    return model


if __name__ == "__main__":
    model = decoder_function()
    dummy_x = torch.rand(2, 512, 16, 16)  # example feature map
    out = model(dummy_x, [torch.rand(2, 256, 32, 32), torch.rand(2, 128, 64, 64), torch.rand(2, 64, 128, 128)])
    print(out.shape)
