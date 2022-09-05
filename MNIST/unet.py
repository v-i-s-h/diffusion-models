from modulefinder import Module
from turtle import forward
from typing import Optional, Tuple, Union, List
import math

import torch
from torch import nn


class Swish(nn.Module):
    """
        Swish activation function
    """
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    """
        Embeddings for $t$
    """
    def __init__(self, n_channels: int):
        """
            `n_channels` is the number of dimensions in the embedding
        """
        super().__init__()
        self.n_channels = n_channels
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        self.act = Swish()
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        # Create sinusoidal position embeddings
        half_dim = self.n_channels // 8
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        # Transform with MLP
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)

        return emb


class ResidualBlock(nn.Module):
    """
        Residual block

        A residual block has two convolution layers with group normalization.
        Each resolution is processed with two residual blocks.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, n_groups: int = 32):
        """
            - `in_channels` is the number of input channels
            - `out_channels` is the number of input channels
            - `time_channels` is the number of channels in the time step ($t$) embeddings
            - `n_groups` is the number of groups for group normalization
        """
        super().__init__()
        # Group normalization and the first convolution layer
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        # Group normalization and second convolution layer
        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = Swish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        # If the number of input channels is not equal to the number of output
        # channels, we have to project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        # Linear layer for time embeddings
        self.time_emb = nn.Linear(time_channels, out_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
            - `x` has shape `[batch_size, in_channels, height, width]`
            - `t` has shape `[batch_size, time_channels]`
        """
        # First convolution layer
        h = self.conv1(self.act1(self.norm1(x)))
        # Add time mebeddings
        h += self.time_emb(t)[:, :, None, None]
        # Second convolution layer
        h = self.conv2(self.act2(self.norm2(h)))

        # Add the shortcut connection and return
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """
        Attention Block

    """
    def __init__(self, n_channels: int, n_heads: int = 1, d_k: int = None, n_groups: int = 32):
        """
            - `n_channels` is the number of channels in the input
            - `n_heads` is the number of heads in multi-head attention
            - `d_k` is the number of dimensions in each head
            - `n_groups` is the number of groups for Group Initialization
        """
        super().__init__()

        # Default `d_k`
        if d_k is None:
            d_k = n_channels
        # Normalization layer
        self.norm = nn.GroupNorm(n_groups, n_channels)
        # Projections for query, key and values
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        # linear layer for final transformation
        self.output = nn.Linear(n_heads * d_k, n_channels)
        # Scale for dot-product attention
        self.scale = d_k ** -0.5
        # 
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
        """
            - `x` has shape `[batch_size, in_channels, height, width]`
            - `t` has shape `[batch_size, time_channels]`
        """
        _ = t
        # Get shape
        batch_size, n_channels, height, width = x.shape
        # Change `x` to shape `[batch_size, seq, n_channels]`
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        # Get query. key and values (concatenated) and shape it to `[batch_size, seq, n_heads, 3 * d_k]`
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        # Split query, key, and values. Each of them will have shape `[batch_size, seq, n_heads, d_k]`
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # Calculate scaled dot-product
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        # Softmax along the sequence dimension
        attn = attn.softmax(dim=1)
        # Multiply by values
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        # Reshape to `[batch_size, seq, n_heads * d_k]`
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        # Transform to `[batch_size, seq, n_channels]`
        res = self.output(res)

        # Add skip connection
        res += x

        # Change to shape `[batch_size, in_channels, height, width]`
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)

        #
        return res


class DownBlock(nn.Module):
    """
        This combines `ResidualBlock` and `AttentionBlock`. These are used in the
        first half of the UNet at each resolution.
    """
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)

        return x


class Downsample(nn.Module):
    """
        Scale down the feature map by half
    """
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, 
                            kernel_size=(3, 3), 
                            stride=(2, 2), 
                            padding=(1, 1))
    
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
            `t` is not used, but it's kept in the arguments because for the attention
            layer function signature to match with `ResidualBlock`
        """
        _ = t
        return self.conv(x)


class MiddleBlock(nn.Module):
    """
        Middle Block

        It combines a `ResidualBlock`, `AttentionBlock`, followed by another `ResidualBlock`.
        This block is applied at the lowest resolution of the U-Net.
    """
    def __init__(self, n_channels: int, time_channels: int):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels)
        self.attn = AttentionBlock(n_channels)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)

        return x


class UpBlock(nn.Module):
    """
        UpBlock

        This combines `ResidualBlock` and `AttentionBlock`. These are used in the
        second half of UNet at each resolution
    """
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output
        # of the same resolution from the first half of the UNet
        self.res = ResidualBlock(in_channels + out_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)

        return x


class UpSample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, 
                        kernel_size=(4, 4), 
                        stride=(2, 2), 
                        padding=(1, 1))
    
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because fot the attention
        # layer function signature to match the `ResidualBlock`
        _ = t
        return self.conv(x)


class UNet(nn.Module):
    """
        U-Net
    """
    def __init__(self, image_channels: int = 3, n_channels: int = 64,
                    ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
                    is_attn: Union[Tuple[bool, ...], List[int]] = (False, False, True, True),
                    n_blocks: int = 2):
        """
            - `image_channels` is the number of channels in the image
            - `n_channels` is number of channels in the initial feature map that we transform the image into
            - `ch_mults` is the list of channel numbers at each resolution. The number of channels is `ch_mults[i] * n_channel`
            - `is_attn` is a list of booleans the indicate whether to use attention at each resolution
            - `n_blocks` is the number of `UpDownBlocks` at each resolution
        """
        super().__init__()

        # Number of resolutions
        n_resolutions = len(ch_mults)

        # Project image into feature map
        self.image_proj = nn.Conv2d(image_channels, n_channels, kernel_size=(3, 3), padding=(1, 1))

        # Time embeddings layer. Time embeddings has `n_channels * 4` channels
        self.time_emb = TimeEmbedding(n_channels * 4)

        # -------------- First half of UNet - decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = n_channels
        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = in_channels * ch_mults[i]
            # Add `n_blocks`
            for _ in range(n_blocks):
                down.append(
                    DownBlock(in_channels, out_channels, n_channels * 4, is_attn[i])
                )
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))
        
        # Combine the set of modules
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(out_channels, n_channels * 4, )

        # -------------- Second half of the UNet - increasing resolution
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(
                    UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i])
                )
            # Final block to reduce the number of channels
            out_channels = in_channels // ch_mults[i]
            up.append(
                UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i])
            )
            in_channels = out_channels
            # up sample at all resolutions except last
            if i > 0:
                up.append(UpSample(in_channels))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        # Final normalization and convolution layer
        self.norm = nn.GroupNorm(8, n_channels)
        self.act = Swish()
        self.final = nn.Conv2d(in_channels, image_channels, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        - `x` has shape `[batch_size, in_channels, height, width]`
        - `t` has shape `[batch_size]`
        """
        # Get time-step embeddings
        t = self.time_emb(t)

        # Get image projection
        x = self.image_proj(x)

        # `h` will store outputs at each resolution for skip connection
        h = [x]
        # First half of UNet
        for m in self.down:
            x = m(x, t)
            h.append(x)

        # Middle
        x = self.middle(x, t)

        # Second half of UNet
        for m in self.up:
            if isinstance(m, UpSample):
                x = m(x, t)
            else:
                # Get the skip connection from first hlaf of UNet and concatenate
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x, t)

        # Final normalization and convolution
        return self.final(self.act(self.norm(x)))
