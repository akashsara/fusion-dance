# Adapted from: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial12/Autoregressive_Image_Modeling.html

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedConvolution(nn.Module):
    def __init__(self, c_in, c_out, mask, **kwargs):
        """
        Implements a convolution with mask applied on its weights.
        Inputs:
            c_in - Number of input channels
            c_out - Number of output channels
            mask - Tensor of shape [kernel_size_H, kernel_size_W] with 0s where
                   the convolution should be masked, and 1s otherwise.
            kwargs - Additional arguments for the convolution
        """
        super().__init__()
        # For simplicity: calculate padding automatically
        kernel_size = (mask.shape[0], mask.shape[1])
        dilation = 1 if "dilation" not in kwargs else kwargs["dilation"]
        padding = tuple([dilation * (kernel_size[i] - 1) // 2 for i in range(2)])
        # Actual convolution
        self.conv = nn.Conv2d(c_in, c_out, kernel_size, padding=padding, **kwargs)

        # Buffers are simply tensors that are a part of the model
        # But are not treated as parameters.
        # Similar to the running mean tracked in batch norm. 
        self.register_buffer("mask", mask[None, None])

    def forward(self, x):
        self.conv.weight.data *= self.mask  # Ensures zero's at masked positions
        return self.conv(x)


class VerticalStackConvolution(MaskedConvolution):
    def __init__(self, c_in, c_out, kernel_size=3, mask_center=False, **kwargs):
        # Mask out all pixels below. For efficiency, we could also reduce the kernel
        # size in height, but for simplicity, we stick with masking here.
        mask = torch.ones(kernel_size, kernel_size)
        mask[kernel_size // 2 + 1 :, :] = 0

        # For the very first convolution, we will also mask the center row
        if mask_center:
            mask[kernel_size // 2, :] = 0

        super().__init__(c_in, c_out, mask, **kwargs)


class HorizontalStackConvolution(MaskedConvolution):
    def __init__(self, c_in, c_out, kernel_size=3, mask_center=False, **kwargs):
        # Mask out all pixels on the left. Note that our kernel has a size of 1
        # in height because we only look at the pixel in the same row.
        mask = torch.ones(1, kernel_size)
        mask[0, kernel_size // 2 + 1 :] = 0

        # For the very first convolution, we will also mask the center pixel
        if mask_center:
            mask[0, kernel_size // 2] = 0

        super().__init__(c_in, c_out, mask, **kwargs)


class GatedMaskedConv(nn.Module):
    def __init__(self, c_in, **kwargs):
        """
        Gated Convolution block implemented the computation graph shown above.
        """
        super().__init__()
        self.conv_vert = VerticalStackConvolution(c_in, c_out=2 * c_in, **kwargs)
        self.conv_horiz = HorizontalStackConvolution(c_in, c_out=2 * c_in, **kwargs)
        self.conv_vert_to_horiz = nn.Conv2d(
            2 * c_in, 2 * c_in, kernel_size=1, padding=0
        )
        self.conv_horiz_1x1 = nn.Conv2d(c_in, c_in, kernel_size=1, padding=0)

    def forward(self, v_stack, h_stack):
        # Vertical stack (left)
        v_stack_feat = self.conv_vert(v_stack)
        v_val, v_gate = v_stack_feat.chunk(2, dim=1)
        v_stack_out = torch.tanh(v_val) * torch.sigmoid(v_gate)

        # Horizontal stack (right)
        h_stack_feat = self.conv_horiz(h_stack)
        h_stack_feat = h_stack_feat + self.conv_vert_to_horiz(v_stack_feat)
        h_val, h_gate = h_stack_feat.chunk(2, dim=1)
        h_stack_feat = torch.tanh(h_val) * torch.sigmoid(h_gate)
        h_stack_out = self.conv_horiz_1x1(h_stack_feat)
        h_stack_out = h_stack_out + h_stack

        return v_stack_out, h_stack_out


class PixelCNN(nn.Module):
    def __init__(self, c_in, c_hidden, num_classes, kernel_size=3):
        super().__init__()
        self.num_classes = num_classes

        # Initial convolutions skipping the center pixel
        self.conv_vstack = VerticalStackConvolution(c_in, c_hidden, kernel_size=kernel_size, mask_center=True)
        self.conv_hstack = HorizontalStackConvolution(c_in, c_hidden, kernel_size=kernel_size, mask_center=True)
        # Convolution block of PixelCNN. We use dilation instead of downscaling
        self.conv_layers = nn.ModuleList(
            [
                GatedMaskedConv(c_hidden, kernel_size=kernel_size),
                GatedMaskedConv(c_hidden, kernel_size=kernel_size, dilation=2),
                GatedMaskedConv(c_hidden, kernel_size=kernel_size),
                GatedMaskedConv(c_hidden, kernel_size=kernel_size, dilation=4),
                GatedMaskedConv(c_hidden, kernel_size=kernel_size),
                GatedMaskedConv(c_hidden, kernel_size=kernel_size, dilation=2),
                GatedMaskedConv(c_hidden, kernel_size=kernel_size),
            ]
        )
        # Output classification convolution (1x1)
        self.conv_out = nn.Conv2d(c_hidden, c_in * self.num_classes, kernel_size=1, padding=0)

    def forward(self, x):
        """
        Forward image through model and return logits for each pixel.
        Inputs:
            x - Image tensor with integer values between 0 and 255.
        """
        # Scale input from 0 to num_classes to -1 to 1
        x = (x.float() / self.num_classes) * 2 - 1

        # Initial convolutions
        v_stack = self.conv_vstack(x)
        h_stack = self.conv_hstack(x)
        # Gated Convolutions
        for layer in self.conv_layers:
            v_stack, h_stack = layer(v_stack, h_stack)
        # 1x1 classification convolution
        # Apply ELU before 1x1 convolution for non-linearity on residual connection
        out = self.conv_out(F.elu(h_stack))

        # Output dimensions: [Batch, Classes, Channels, Height, Width]
        out = out.reshape(
            out.shape[0], self.num_classes, out.shape[1] // self.num_classes, out.shape[2], out.shape[3]
        )
        return out

    def calc_likelihood(self, x):
        """
        bits per dimension (bpd) is motivated from an information theory 
        perspective and describes how many bits we would need to encode a
        particular example in our modeled distribution. The less bits we need,
        the more likely the example in our distribution. When we test for the
        bits per dimension of our test dataset, we can judge whether our model
        generalizes to new samples of the dataset and didn’t memorize the
        training dataset.
        """
        # Forward pass with bpd likelihood calculation
        pred = self.forward(x)
        nll = F.cross_entropy(pred, x, reduction='none')
        bpd = nll.mean(dim=[1,2,3]) * np.log2(np.exp(1))
        return bpd.mean()

    @torch.no_grad()
    def sample(self, image_shape, device, img=None):
        """
        Sampling function for the autoregressive model.
        Inputs:
            img_shape - Shape of the image to generate (B,C,H,W)
            img (optional) - If given, this tensor will be used as
                             a starting image. The pixels to fill
                             should be -1 in the input tensor.
        """
        # Create empty image
        if img is None:
            img = torch.zeros(image_shape, dtype=torch.long).to(device) - 1
        # Generation loop
        for h in tqdm(range(image_shape[2]), leave=False):
            for w in range(image_shape[3]):
                for c in range(image_shape[1]):
                    # Skip if not to be filled (-1)
                    if (img[:, c, h, w] != -1).all().item():
                        continue
                    # For efficiency, we only have to input the upper part of the image
                    # as all other parts will be skipped by the masked convolutions anyways
                    pred = self.forward(img[:, :, : h + 1, :])
                    probs = F.softmax(pred[:, :, c, h, w], dim=-1)
                    img[:, c, h, w] = torch.multinomial(probs, num_samples=1).squeeze(
                        dim=-1
                    )
        return img
