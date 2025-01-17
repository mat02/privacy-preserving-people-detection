# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Block modules
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ultralytics.yolo.utils import LOGGER, TryExcept, plt_settings, threaded

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, CBAM, SqueezeExcitation
from .transformer import TransformerBlock

__all__ = ('DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3', 'C2f', 'C3x', 'C3TR', 'C3Ghost',
           'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'Proto', 'RepC3', 'MP', 'SP', 'SPF',
           'StemBlockAttention', 'Shuffle_Block_Attention', 'GhostDWConvblock', 'AdaptationBlock',
           'Shuffle_Block_SE', 'Shuffle_Block_Xception')

import math

class OriginalLinear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['bias']

    def __init__(self, in_features, out_features, bias=False):
        super(OriginalLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        #self.weight = torch.abs(self.weight)
    def reset_parameters(self):
        # sample from uniform distribution
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            nn.init.uniform_(self.bias,-bound, bound)
        #self.weight = torch.abs(self.weight)
 #   @weak_script_method
    def forward(self, input):
      weight = self.weight
      if input.dim() == 2 and self.bias is not None:
        # fused op is marginally faster
        #weight = F.softmax(F.softmax(F.relu(self.weight),dim=0),dim=1)
        ret = torch.addmm(self.bias, input,weight)
      else:
        output = input.matmul(weight)
        if self.bias is not None:
            output += self.bias
        ret = output
      return ret,weight
#      return ret,F.relu(self.weight)
      #  return linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class AdaptationBlock(nn.Module):
    def __init__(self, c1, c2, k=4, num_blocks=8, dummy=None, unique_blocks=False):
        super().__init__()

        self.k = k
        self.c1 = c1
        self.c2 = c2
        self.b = num_blocks
        self.dummy = dummy
        self.unique_blocks = unique_blocks
        
        self.vis_fm = 0

        # ps - pixel scrambling
        if not self.unique_blocks:
            self.ps_conv = nn.Conv2d(c1, c2, kernel_size=k, stride=k, padding=0, bias=False)
            self.ps_bn = nn.BatchNorm2d(c2)
        else:
            self.ps_conv = nn.ModuleList([nn.Conv2d(c1, c2, kernel_size=k, stride=k, padding=0, bias=False) for _ in range(self.b * self.b)])
            self.ps_bn = nn.ModuleList([nn.BatchNorm2d(c2) for _ in range(self.b * self.b)])
            
        self.ps_act = nn.LeakyReLU(0.2)

        # bs - block scrambling
        self.bs_matrix = OriginalLinear(self.b * self.b, self.b * self.b)
        self.bs_pixel_shuffle = nn.PixelShuffle(k)

        if self.dummy:
            if self.dummy.lower() == 'identity':
                self.generate_identity_adaptation()
            elif self.dummy.lower() == 'fixed':
                self.generate_fixed_adaptation()
            else:
                raise ValueError(f'Unknown value for `dummy` parameter: {self.dummy}')
            
    def visualize_adaptation_network(self, epoch, save_dir, n=16):
        """
        Visualize feature maps of a given model module during inference.

        Args:
            i (int): Epoch
            save_dir (Path): Directory to save results.
            n (int, optional): Maximum number of feature maps to plot. Defaults to 32.
        """

        # Pixel permutation matrix
        save_dir.mkdir(parents=True, exist_ok=True)
        f = save_dir / f"adaptnet_pixel_mtx_{epoch}.png"  # filename

        ps_conv_data = self.ps_conv.weight.data.cpu() if not isinstance(self.ps_conv, nn.ModuleList) else self.ps_conv[0].weight.data.cpu()
        blocks = torch.chunk(ps_conv_data, self.c2, dim=0)  # select batch index 0, block by channels
        n = min(n, self.c2)  # number of plots
        fig, ax = plt.subplots(math.ceil(n / 8), 8 * 3, tight_layout=True)  # 8 * 3 cols x n/8 rows
        ax = ax.ravel()
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        for i in range(n):
            for k in range(3):
                ax[i * 3 + k].imshow(blocks[i].squeeze()[k], cmap='viridis')  # cmap='gray'
                ax[i * 3 + k].axis('off')

        plt.savefig(f, dpi=300, bbox_inches='tight')
        plt.close(fig)

        # Block permutation matrix
        f = save_dir / f"adaptnet_block_mtx_{epoch}.png"  # filename

        fig = plt.figure()
        plt.imshow(self.bs_matrix.weight.data.cpu(), figure=fig, cmap='viridis')

        plt.savefig(f, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def visualize_tensor(self, x, idx=None, name='vis'):
        from matplotlib import pyplot as plt
        from skimage.exposure import rescale_intensity
        x = x.detach().cpu().numpy()
        if idx is not None:
            x = x[idx]
        if len(x.shape) > 2:
            x = x.transpose((1, 2, 0))
        x = x.astype('float32')
        for i in range(x.shape[-1]):  # Iterate over each channel
            x[:, :, i] = rescale_intensity(
                x[:, :, i], in_range='image', out_range=(0, 1)
            )
        plt.axis('off')
        fig = plt.figure()
        plt.imshow(x[:, :, :3], figure=fig)
        fig.axes[0].get_xaxis().set_visible(False)
        fig.axes[0].get_yaxis().set_visible(False)
        fig.savefig(f'{name}.png', bbox_inches='tight', pad_inches = 0)
        plt.close(fig)

    def generate_identity_adaptation(self):
        with torch.no_grad():
            ps_identity_tensors = []
            for _ in range(self.c2 // (self.k * self.k)):
                for i in range(self.k):
                    for j in range(self.k):
                        t = torch.zeros([self.c1, self.k, self.k])
                        t[0, i, j] = 1.0
                        ps_identity_tensors.append(t)
            
            # for _ in range(self.c2):
            #     t = torch.zeros([self.c1, self.k, self.k])
            #     t[0] = torch.eye(self.k)
            #     ps_identity_tensors.append(t)

            if isinstance(self.ps_conv, nn.ModuleList):
                for ps_conv in self.ps_conv:
                    ps_conv.weight.data = torch.stack(ps_identity_tensors)
            else:
                self.ps_conv.weight.data = torch.stack(ps_identity_tensors)

            self.bs_matrix.weight.data = torch.eye(self.b * self.b)
            self.requires_grad_(False)
            self.ps_conv.requires_grad_(False)
            self.bs_matrix.requires_grad_(False)
            

    def forward(self, x):
        bn, c_in, h_in, w_in = x.size()
        
        if self.unique_blocks:
            x = torch.permute(x, (0, 2, 3, 1)) # change channel ordering to channels-last
            x = torch.reshape(x, (bn, h_in // (self.k * self.b), self.b, self.k, w_in // (self.k * self.b), self.b, self.k, c_in)) # reshape tensor
            x = torch.permute(x, (0, 1, 4, 2, 5, 3, 6, 7)) # permute tensor to batch, patches, blocks, pixels, channels
            x = torch.reshape(x, (bn, -1, self.k, self.k, c_in)) # flatten blocks
            
            out_blocks = None
            num_patches = h_in // (self.k * self.b), w_in // (self.k * self.b)
            for layer_id in range(len(self.ps_conv)):
                block_ids = torch.arange(0 + layer_id, x.size()[1], len(self.ps_conv))
                blocks_for_layer = x[:, block_ids]
                blocks_for_layer = torch.reshape(blocks_for_layer, (bn, num_patches[0], num_patches[1], self.k, self.k, c_in))
                blocks_for_layer = torch.permute(blocks_for_layer, (0, 1, 3, 2, 4, 5))
                blocks_for_layer = torch.reshape(blocks_for_layer, (bn, num_patches[0] * self.k, num_patches[1] * self.k, c_in))
                blocks_for_layer = torch.permute(blocks_for_layer, (0, 3, 1, 2)) # permute tensor to channels-first
                
                res = self.ps_conv[layer_id](blocks_for_layer)
                res = self.ps_act(self.ps_bn[layer_id](res)) if bn > 1 else self.ps_act(res)
                
                res_for_layer = torch.reshape(res, (bn, self.c2, num_patches[0] * num_patches[1]))
                
                if out_blocks is None:
                    out_blocks = torch.zeros((bn, self.c2, x.shape[1]), dtype=res_for_layer.dtype, device=x.device)
                out_blocks[:, :, block_ids] = res_for_layer
            
            x = torch.reshape(out_blocks, (bn, self.c2, num_patches[0], num_patches[1], self.b, self.b)) # reshape tensor bn, c2, patch row, patch col, block row, block col
            x = torch.permute(x, (0, 1, 2, 4, 3, 5)) # reorder
            x = torch.reshape(x, (bn, self.c2, h_in // self.k, w_in // self.k)) # reshape tensor to move blocks and channels to single dim
        else:
            # pixel scrambling
            x = self.ps_conv(x)
            if not self.dummy:
                x = self.ps_bn(x)
                x = self.ps_act(x)

        # reshape to 1d
        #b, c, w // 4, h // 4
        #b, n, c, w // (4*8), h // (4*8)
        #b, n, c, 64
        bn, c_ps, h_ps, w_ps = x.size()
        

        x = torch.permute(x, (0, 2, 3, 1)) # change channel ordering to channels-last
        x = torch.reshape(x, (bn, h_ps // self.b, self.b, w_ps // self.b, self.b, c_ps)) # reshape tensor
        x = torch.permute(x, (0, 1, 3, 5, 2, 4)) # permute tensor to channels-first, bn, hs_ps // self.b, w_ps // self.b, c_ps, self.b, self.b
        x = torch.reshape(x, (-1, self.b * self.b)) # reshape tensor to move blocks and channels to single dim
        
        x, permutation_mtx = self.bs_matrix(x)

        x = torch.reshape(x, (bn, h_ps // self.b, w_ps // self.b, c_ps, self.b, self.b)) # reshape tensor to original dimensions
        x = torch.permute(x, (0, 1, 4, 2, 5, 3)) # change channel ordering to channels-last
        x = torch.reshape(x, (bn, h_ps, w_ps, c_ps)) # reshape tensor to move blocks and channels to single dim
        x = torch.permute(x, (0, 3, 1, 2)) # permute tensor to channels-first

        x = self.bs_pixel_shuffle(x)

        return x, permutation_mtx

class Split(nn.Module):
    def __init__(self, idx=0):
        super().__init__()

        self.idx = idx

    def forward(self, x):
        if not isinstance(x, (list, tuple)):
            print("Warning! Split layer received input that is not a list")
            return x
        if len(x) < self.idx:
            raise IndexError(f"Split layer received input list that has less elements than index ({len(x)}, {self.idx})")
        return x[self.idx]


class MP(nn.Module):
    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)

class SP(nn.Module):
    def __init__(self, k=3, s=1):
        super(SP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=s, padding=k // 2)

    def forward(self, x):
        return self.m(x)

class SPF(nn.Module):
    def __init__(self, k=3, s=1):
        super(SPF, self).__init__()
        self.n = (k - 1) // 2
        self.m = nn.Sequential(*[nn.MaxPool2d(kernel_size=3, stride=s, padding=1) for _ in range(self.n)])

    def forward(self, x):
        return self.m(x)

# yolov7-lite
class StemBlock(nn.Module):
    def __init__(self, c1, c2, k=3, s=2, p=None, g=1, d=1, act=True):
        super(StemBlock, self).__init__()
        self.stem_1 = Conv(c1, c2, k, s, p, g, d, act)
        self.stem_2a = Conv(c2, c2 // 2, 1, 1, 0)
        self.stem_2b = Conv(c2 // 2, c2, 3, 2, 1)
        self.stem_2p = nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True)
        self.stem_3 = Conv(c2 * 2, c2, 1, 1, 0)

    def forward(self, x):
        stem_1_out  = self.stem_1(x)
        stem_2a_out = self.stem_2a(stem_1_out)
        stem_2b_out = self.stem_2b(stem_2a_out)
        stem_2p_out = self.stem_2p(stem_1_out)
        out = self.stem_3(torch.cat((stem_2b_out,stem_2p_out),1))
        return out

class StemBlockAttention(nn.Module):
    def __init__(self, c1, c2, k=3, s=2, p=None, g=1, d=1, act=True):
        super(StemBlockAttention, self).__init__()
        self.stem_1 = Conv(c1, c2, k, s, p, g, d, act)
        self.stem_2a = Conv(c2, c2 // 2, 1, 1, 0)
        self.stem_2b = Conv(c2 // 2, c2, 3, 2, 1)
        self.stem_2p = nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True)
        self.stem_3 = Conv(c2 * 2, c2, 1, 1, 0)

        self.cbam = CBAM(c2, kernel_size=3)


    def forward(self, x):
        stem_1_out  = self.stem_1(x)

        # stem_2a_out = self.stem_2a(stem_1_out)
        stem_2a_out = self.stem_2a(stem_1_out)
        stem_2b_out = self.stem_2b(stem_2a_out)
        stem_2p_out = self.stem_2p(stem_1_out)
        stem_2b_out = self.cbam(stem_2b_out)

        out = self.stem_3(torch.cat((stem_2b_out,stem_2p_out),1))
        return out

class conv_bn_relu_maxpool(nn.Module):
    def __init__(self, c1, c2):  # ch_in, ch_out
        super(conv_bn_relu_maxpool, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

    def forward(self, x):
        return self.maxpool(self.conv(x))

class DWConvblock(nn.Module):
    "Depthwise conv + Pointwise conv"
    def __init__(self, in_channels, out_channels, k, s):
        super(DWConvblock, self).__init__()
        self.p = k // 2
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=k, stride=s, padding=self.p, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x

class GhostDWConvblock(nn.Module):
    "Ghost Depthwise conv + Pointwise conv"
    def __init__(self, in_channels, k, s):
        super(GhostDWConvblock, self).__init__()
        self.p = k // 2

        self.c_ = in_channels // 2
        self.conv1 = nn.Conv2d(in_channels, self.c_, kernel_size=k, stride=s, padding=self.p, groups=self.c_, bias=False)
        self.bn1 = nn.BatchNorm2d(self.c_)
        self.act1 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(self.c_, self.c_, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(self.c_)
        self.act2 = nn.SiLU(inplace=True)

    def forward(self, x):
        # Normal conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        # print('x', x.size())

        # DW Conv
        y = self.conv2(x)
        y = self.bn2(y)
        y = self.act2(y)
        # print('y', y.size())

        out = channel_shuffle(torch.cat((x, y), dim=1), 2)
        # print('out', out.size())
        return out

class ADD(nn.Module):
    # Stortcut a list of tensors along dimension
    def __init__(self, alpha=0.5):
        super(ADD, self).__init__()
        self.a = alpha

    def forward(self, x):
        x1, x2 = x[0], x[1]
        return torch.add(x1, x2, alpha=self.a)

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

class Shuffle_Block(nn.Module):
    def __init__(self, inp, oup, stride):
        super(Shuffle_Block, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.SiLU(inplace=True),
            )

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.SiLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.SiLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out

class Shuffle_Block_Attention(nn.Module):
    def __init__(self, inp, oup, stride):
        super(Shuffle_Block_Attention, self).__init__()
        self.sb = Shuffle_Block(inp, oup, stride)
        self.cbam = CBAM(oup, kernel_size=7)

    def forward(self, x):
        out = self.sb(x)
        out = self.cbam(out)
        return out

# end of yolov7-lite

class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).
    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):  # ch_in, number of protos, number of masks
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """HG_Block of PPHGNetV2 with 2 convolutions and LightConv.
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):  # ch_in, ch_out, number
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1,
                                                                            act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))
