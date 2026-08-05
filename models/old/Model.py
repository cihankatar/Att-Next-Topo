import torch
import torch.nn as nn
import torch.nn.functional as F
from models.enc import encoder_function
from models.dec import decoder_function


def get_device():
    """Return the appropriate device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Head(nn.Module):
    """Prediction head with depthwise convolution, normalization, and linear projections."""

    def __init__(self, in_channels=64, out_channels=1):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding="same", groups=in_channels)
        self.norm = nn.LayerNorm(in_channels)
        self.act = nn.GELU()
        self.pwconv1 = nn.Linear(in_channels, in_channels)
        self.pwconv2 = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        # Depthwise convolution
        x = self.depthwise_conv(x)
        x = self.norm(x.permute(0, 2, 3, 1))  # Convert to (B, H, W, C) for LayerNorm
        x = self.act(x)

        # Pointwise projections
        x = self.pwconv1(x)
        x = self.norm(x)
        x = self.act(x)

        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # Back to (B, C, H, W)
        return x

class Bottleneck(nn.Module):
    """Bottleneck block with depthwise + pointwise linear layers and residual connection."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        hidden_channels = int(4 * in_channels)

        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding="same", groups=in_channels)
        self.pwconv1 = nn.Linear(in_channels, hidden_channels)
        self.pwconv2 = nn.Linear(hidden_channels, out_channels)
        self.norm = nn.LayerNorm(in_channels)
        self.act = nn.GELU()

        # Optionally, you could add attention here if needed
        # self.attention = Attention(dim=in_channels)

    def forward(self, x):
        residual = x
        x = self.depthwise_conv(x).permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x).permute(0, 3, 1, 2)  # Back to (B, C, H, W)
        return x + residual


class ATTNext(nn.Module):
    """Main Encoder-Decoder network with bottleneck and head."""

    def __init__(self, training_mode="ssl"):
        super().__init__()
        self.training_mode = training_mode

        self.encoder = encoder_function()
        if self.training_mode == "ssl_pretrained":
            for param in self.encoder.parameters():
                param.requires_grad = False  # Freeze encoder weights

        self.bottleneck = Bottleneck(512, 512)
        self.decoder = decoder_function()
        self.head = Head(in_channels=64, out_channels=1)

    def forward(self, x):
        # Encoder
        if self.training_mode == "ssl_pretrained":
            encoder_features = x  # Directly use provided features
        else:
            encoder_features = self.encoder(x)

        skip_connections = encoder_features[:3][::-1]  # Reverse first 3 features for decoder
        bottleneck_out = self.bottleneck(encoder_features[3])

        # Decoder 
        out = self.decoder(bottleneck_out, skip_connections)

        #Segmentation head
        out = self.head(out)
        return out


if __name__ == "__main__":
    x = torch.randn((2, 3, 256, 256))
    model = ATTNext()
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
