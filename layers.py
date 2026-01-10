import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as wn
import torch.distributions as D

# Basic Layers 
# -------------------------------------------------------------------------------------------------------

def get_linear_ar_mask(n_in, n_out, zerodiagonal=False):
    """Create channel-wise autoregressive mask matching OpenAI's implementation."""
    assert n_in % n_out == 0 or n_out % n_in == 0, f"{n_in} - {n_out}"
    
    mask = np.ones([n_in, n_out], dtype=np.float32)
    if n_out >= n_in:
        k = n_out // n_in
        for i in range(n_in):
            mask[i + 1:, i * k:(i + 1) * k] = 0
            if zerodiagonal:
                mask[i:i + 1, i * k:(i + 1) * k] = 0
    else:
        k = n_in // n_out
        for i in range(n_out):
            mask[(i + 1) * k:, i:i + 1] = 0
            if zerodiagonal:
                mask[i * k:(i + 1) * k:, i:i + 1] = 0
    return mask


def get_conv_ar_mask(h, w, n_in, n_out, zerodiagonal=False):
    """Create convolutional autoregressive mask matching OpenAI's implementation."""
    l = (h - 1) // 2
    m = (w - 1) // 2
    mask = np.ones([h, w, n_in, n_out], dtype=np.float32)
    mask[:l, :, :, :] = 0
    mask[l, :m, :, :] = 0
    # Apply channel-wise autoregressive mask at center position
    mask[l, m, :, :] = get_linear_ar_mask(n_in, n_out, zerodiagonal)
    return mask


class ARConv2d(nn.Conv2d):
    """Channel-wise autoregressive convolution matching OpenAI's implementation."""
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, padding=1, zerodiagonal=True, **kwargs):
        super(ARConv2d, self).__init__(n_in, n_out, kernel_size, stride, padding, **kwargs)
        self.zerodiagonal = zerodiagonal
        
        # Create autoregressive mask
        mask = get_conv_ar_mask(kernel_size, kernel_size, n_in, n_out, zerodiagonal)
        # Convert from [h, w, in, out] to PyTorch format [out, in, h, w]
        mask = torch.from_numpy(mask).permute(3, 2, 0, 1).float()
        self.register_buffer('mask', mask)
        
        # Initialize weights with mask applied
        with torch.no_grad():
            self.weight.data = self.weight.data * self.mask
        
    def forward(self, x):
        # Apply mask to weights (works with weight normalization wrapper)
        # For weight normalization, this masks the computed weight
        # The mask is applied each forward pass to ensure it's always active
        weight = self.weight * self.mask
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class ARMultiConv2d(nn.Module):
    """Autoregressive multi-layer convolution matching OpenAI's ar_multiconv2d."""
    def __init__(self, n_h, n_out, args, nl=F.elu):
        super(ARMultiConv2d, self).__init__()
        self.nl = nl

        convs, out_convs = [], []

        # Hidden layers: zerodiagonal=False (can see current position)
        for i, size in enumerate(n_h):
            n_in = args.z_size if i == 0 else args.h_size
            convs += [wn(ARConv2d(n_in, args.h_size, 3, 1, 1, zerodiagonal=False))]
        
        # Output layers: zerodiagonal=True (cannot see current position)
        for i, size in enumerate(n_out):
            out_convs += [wn(ARConv2d(args.h_size, args.z_size, 3, 1, 1, zerodiagonal=True))]

        self.convs = nn.ModuleList(convs)
        self.out_convs = nn.ModuleList(out_convs)

    def forward(self, x, context):
        # Process through hidden layers
        for i, conv_layer in enumerate(self.convs):
            x = conv_layer(x)
            if i == 0: 
                x += context  # Add context to first layer
            x = self.nl(x)

        # Output layers
        return [conv_layer(x) for conv_layer in self.out_convs]


# IAF building block
# -------------------------------------------------------------------------------------------------------

class IAFLayer(nn.Module):
    def __init__(self, args, downsample):
        super(IAFLayer, self).__init__()
        n_in  = args.h_size
        n_out = args.h_size * 2 + args.z_size * 2
        
        self.z_size = args.z_size
        self.h_size = args.h_size
        self.iaf    = args.iaf
        self.ds     = downsample
        self.args   = args

        if downsample:
            stride, padding, filter_size = 2, 1, 4
            self.down_conv_b = wn(nn.ConvTranspose2d(args.h_size + args.z_size, args.h_size, 4, 2, 1))
        else:
            stride, padding, filter_size = 1, 1, 3
            self.down_conv_b = wn(nn.Conv2d(args.h_size + args.z_size, args.h_size, 3, 1, 1))

        # create modules for UP pass: 
        # up_conv1: outputs [qz_mean, qz_logsd, up_context, h]
        self.up_conv_a = wn(nn.Conv2d(n_in, n_out, filter_size, stride, padding))
        # up_conv3: processes h (matching OpenAI's up_conv3)
        self.up_conv_b = wn(nn.Conv2d(args.h_size, args.h_size, 3, 1, 1))

        # create modules for DOWN pass: 
        self.down_conv_a  = wn(nn.Conv2d(n_in, 4 * self.z_size + 2 * self.h_size, 3, 1, 1))

        if args.iaf:
            self.down_ar_conv = ARMultiConv2d([args.h_size] * 2, [args.z_size] * 2, args)


    def up(self, input):
        x = F.elu(input)
        out_conv = self.up_conv_a(x)
        self.qz_mean, self.qz_logsd, self.up_context, h = out_conv.split([self.z_size] * 2 + [self.h_size] * 2, 1)

        h = F.elu(h)
        h = self.up_conv_b(h)

        if self.ds:
            # F.upsample is deprecated, use F.interpolate instead
            input = F.interpolate(input, scale_factor=0.5, mode='nearest')

        return input + 0.1 * h
        

    def down(self, input, sample=False):
        x = F.elu(input)
        x = self.down_conv_a(x)
        
        pz_mean, pz_logsd, rz_mean, rz_logsd, down_context, h_det = x.split([self.z_size] * 4 + [self.h_size] * 2, 1)
        # In TensorFlow: DiagonalGaussian(mean, 2*logsd) means logvar = 2*logsd, so std = exp(logsd)
        # PyTorch Normal takes std directly, so we use exp(logsd) not exp(2*logsd)
        prior = D.Normal(pz_mean, torch.exp(pz_logsd))
            
        if sample:
            z = prior.rsample()
            kl = kl_obj = torch.zeros(input.size(0)).to(input.device)
        else:
            # In TensorFlow: DiagonalGaussian(mean, 2*(rz_logsd + qz_logsd)) means std = exp(rz_logsd + qz_logsd)
            posterior = D.Normal(rz_mean + self.qz_mean, torch.exp(rz_logsd + self.qz_logsd))
            
            z = posterior.rsample()
            logqs = posterior.log_prob(z) 
            context = self.up_context + down_context

            if self.iaf:
                x = self.down_ar_conv(z, context) 
                arw_mean, arw_logsd = x[0] * 0.1, x[1] * 0.1
                z = (z - arw_mean) / torch.exp(arw_logsd)
            
                # the density at the new point is the old one + determinant of transformation
                logq = logqs
                logqs += arw_logsd

            logps = prior.log_prob(z) 
            kl = logqs - logps

            # free bits (doing as in the original repo, even if weird)
            kl_obj = kl.sum(dim=(-2, -1)).mean(dim=0, keepdim=True)
            kl_obj = kl_obj.clamp(min=self.args.free_bits)
            kl_obj = kl_obj.expand(kl.size(0), -1)
            kl_obj = kl_obj.sum(dim=1)

            # sum over all the dimensions, but the batch
            kl = kl.sum(dim=(1,2,3))

        h = torch.cat((z, h_det), 1)
        h = F.elu(h)

        if self.ds:
            # F.upsample is deprecated, use F.interpolate instead
            input = F.interpolate(input, scale_factor=2., mode='nearest')
        
        h = self.down_conv_b(h)

        return input + 0.1 * h, kl, kl_obj 

