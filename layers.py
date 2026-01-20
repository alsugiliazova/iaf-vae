import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as wn
import torch.distributions as D

# Basic Layers 
# -------------------------------------------------------------------------------------------------------

# taken from https://github.com/jzbontar/pixelcnn-pytorch
class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)

class ARMultiConv2d(nn.Module):
    def __init__(self, n_h, n_out, args, nl=F.elu):
        super(ARMultiConv2d, self).__init__()
        self.nl = nl

        convs, out_convs = [], []

        for i, size in enumerate(n_h):
            convs     += [MaskedConv2d('A' if i == 0 else 'B', args.z_size if i == 0 else args.h_size, args.h_size, 3, 1, 1)]
        for i, size in enumerate(n_out):
            out_convs += [MaskedConv2d('B', args.h_size, args.z_size, 3, 1, 1)]

        self.convs = nn.ModuleList(convs)
        self.out_convs = nn.ModuleList(out_convs)


    def forward(self, x, context):
        for i, conv_layer in enumerate(self.convs):
            x = conv_layer(x)
            if i == 0: 
                x += context
            x = self.nl(x)

        return [conv_layer(x) for conv_layer in self.out_convs]


# Autoregressive Prior (MADE-style)
# -------------------------------------------------------------------------------------------------------

class MaskedLinear(nn.Linear):
    """Masked linear layer for MADE-style autoregressive networks"""
    def __init__(self, in_features, out_features, mask, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        # Ensure mask has correct shape: (out_features, in_features) to match weight
        if mask.shape != (out_features, in_features):
            raise ValueError(f"Mask shape {mask.shape} doesn't match expected ({out_features}, {in_features})")
        self.register_buffer('mask', mask)
    
    def forward(self, input):
        # Weight has shape (out_features, in_features), mask should match
        masked_weight = self.weight * self.mask
        return F.linear(input, masked_weight, self.bias)


def create_masks(n_in, n_out, n_hidden, n_layers, input_order='sequential', output_order='sequential'):
    """
    Create masks for MADE-style autoregressive network.
    
    Ensures that output dimension i only depends on input dimensions < i.
    Uses degree-based masking: each unit has a degree, and connections are
    allowed only from lower-degree to higher-degree units.
    
    Args:
        n_in: Input dimension
        n_out: Output dimension (should be 2 * n_in for mean and log_std)
        n_hidden: Hidden dimension
        n_layers: Number of hidden layers
        input_order: Ordering of input dimensions (not used, kept for compatibility)
        output_order: Ordering of output dimensions (not used, kept for compatibility)
    
    Returns:
        List of masks for each layer
    """
    masks = []
    # Input degrees: dimension i has degree i (1-indexed: 1, 2, ..., n_in)
    input_degrees = torch.arange(1, n_in + 1)
    
    # Create degrees for each layer
    degrees = [input_degrees]
    for i in range(n_layers + 1):
        if i == n_layers:
            # Output layer: n_out should be 2 * n_in (mean and log_std for each dimension)
            assert n_out == 2 * n_in, f"Output dimension must be 2 * input dimension, got {n_out} != 2 * {n_in}"
            # Output dimension i (for both mean and log_std) should have degree i+1
            # This ensures it can depend on inputs with degree <= i (i.e., inputs < i+1)
            # But we want it to depend on inputs < i, so we need to be careful
            # Actually, if output i has degree i+1, it can connect to inputs with degree < i+1, i.e., <= i
            # But we want it to connect to inputs < i, so we need output i to have degree i
            # Wait, let's think: if output i has degree i, it can connect to inputs with degree < i, i.e., <= i-1
            # That's exactly what we want! So output i should have degree i.
            output_degrees = torch.arange(1, n_in + 1).repeat(2)
            degrees.append(output_degrees)
        else:
            # Hidden layer: assign degrees randomly between 1 and n_in (inclusive)
            # This ensures good connectivity while maintaining autoregressive property
            degrees.append(torch.randint(1, n_in + 1, (n_hidden,)))
    
    # Create masks: connection from unit j (degree d_j) to unit i (degree d_i) is allowed if d_i > d_j
    for i in range(len(degrees) - 1):
        in_deg = degrees[i]  # Shape: (in_dim,)
        out_deg = degrees[i + 1]  # Shape: (out_dim,)
        
        # Expand for broadcasting: (out_dim, 1) and (1, in_dim) -> (out_dim, in_dim)
        in_degrees = in_deg.unsqueeze(0)  # (1, in_dim)
        out_degrees = out_deg.unsqueeze(1)  # (out_dim, 1)
        
        # Allow connection if output degree > input degree
        # Broadcasting: (out_dim, 1) > (1, in_dim) -> (out_dim, in_dim)
        mask = (out_degrees > in_degrees).float()
        
        # Verify mask shape: should be (out_features, in_features) to match weight matrix
        expected_out = len(out_deg)
        expected_in = len(in_deg)
        if mask.shape != (expected_out, expected_in):
            raise ValueError(f"Mask shape mismatch: got {mask.shape}, expected ({expected_out}, {expected_in})")
        masks.append(mask)
    
    return masks


class AutoregressivePrior(nn.Module):
    """
    Autoregressive Gaussian prior p(z) = ∏i N(z_i | μ_i(z_<i), σ_i(z_<i))
    
    Uses MADE-style masked networks to ensure autoregressive structure.
    The masks ensure that output i only depends on inputs < i, allowing
    parallel computation of all means/stds.
    """
    def __init__(self, z_dim, hidden_dim=128, n_layers=2):
        """
        Args:
            z_dim: Total dimension of z (z_size * H * W after flattening)
            hidden_dim: Hidden dimension for MLP
            n_layers: Number of hidden layers
        """
        super(AutoregressivePrior, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # Create masks for autoregressive structure
        masks = create_masks(z_dim, 2 * z_dim, hidden_dim, n_layers)
        
        # Verify we have the right number of masks
        expected_num_masks = n_layers + 1  # input->hidden, hidden->hidden (if any), hidden->output
        if len(masks) != expected_num_masks:
            raise ValueError(f"Expected {expected_num_masks} masks, got {len(masks)}")
        
        # Build network
        layers = []
        # Input layer: z_dim -> hidden_dim
        mask0 = masks[0]
        if mask0.shape != (hidden_dim, z_dim):
            raise ValueError(f"First mask shape {mask0.shape} doesn't match expected ({hidden_dim}, {z_dim})")
        input_layer = MaskedLinear(z_dim, hidden_dim, mask0)
        # Initialize input layer with small weights
        with torch.no_grad():
            input_layer.weight.data *= 0.1
            input_layer.bias.data.zero_()
        layers.append(input_layer)
        layers.append(nn.ELU())
        
        # Hidden layers: hidden_dim -> hidden_dim
        for i in range(1, n_layers):
            mask_i = masks[i]
            if mask_i.shape != (hidden_dim, hidden_dim):
                raise ValueError(f"Hidden mask {i} shape {mask_i.shape} doesn't match expected ({hidden_dim}, {hidden_dim})")
            hidden_layer = MaskedLinear(hidden_dim, hidden_dim, mask_i)
            # Initialize hidden layers with small weights
            with torch.no_grad():
                hidden_layer.weight.data *= 0.1
                hidden_layer.bias.data.zero_()
            layers.append(hidden_layer)
            layers.append(nn.ELU())
        
        # Output layer: hidden_dim -> 2*z_dim (mean and log_std for each dimension)
        mask_out = masks[-1]
        if mask_out.shape != (2 * z_dim, hidden_dim):
            raise ValueError(f"Output mask shape {mask_out.shape} doesn't match expected ({2 * z_dim}, {hidden_dim})")
        output_layer = MaskedLinear(hidden_dim, 2 * z_dim, mask_out)
        
        # Initialize output layer to produce reasonable priors
        # Initialize means to ~0 and log_stds to ~0 (std ~1) to match standard Normal
        with torch.no_grad():
            # Means: initialize to small values near 0
            output_layer.weight.data[:z_dim, :] *= 0.01
            output_layer.bias.data[:z_dim].zero_()
            # Log_stds: initialize to small negative values (std slightly < 1)
            output_layer.weight.data[z_dim:, :] *= 0.01
            output_layer.bias.data[z_dim:].fill_(-0.5)  # log_std ≈ -0.5 means std ≈ 0.6
        
        layers.append(output_layer)
        self.net = nn.Sequential(*layers)
    
    def forward(self, z_flat):
        """
        Compute log p(z) for autoregressive prior.
        
        Args:
            z_flat: Flattened z of shape (B, z_dim) where z_dim = z_size * H * W
        
        Returns:
            log_prob: Log probability of shape (B,)
            means: Means for each dimension (B, z_dim)
            log_stds: Log standard deviations for each dimension (B, z_dim)
        """
        B = z_flat.size(0)
        z_dim = z_flat.size(1)
        
        if z_dim != self.z_dim:
            raise ValueError(f"Expected z_dim={self.z_dim}, got {z_dim}")
        
        # Forward through masked network
        # The masks ensure that output i only depends on z_<i
        output = self.net(z_flat)
        
        # Split into means and log_stds
        means = output[:, :z_dim]  # (B, z_dim)
        log_stds = output[:, z_dim:]  # (B, z_dim)
        
        # Clamp log_std for numerical stability
        # Use reasonable bounds: std between ~0.01 and ~7.4
        log_stds = torch.clamp(log_stds, min=-4.6, max=2.0)
        
        # Compute standard deviations
        stds = torch.exp(log_stds)
        
        # Ensure minimum std for numerical stability
        stds = torch.clamp(stds, min=1e-6)
        
        # Compute log probability: log p(z) = Σ_i log N(z_i | μ_i, σ_i)
        # PyTorch's Normal.log_prob already includes the normalization term
        dist = D.Normal(means, stds)
        log_prob = dist.log_prob(z_flat).sum(dim=1)  # Sum over dimensions
        
        return log_prob, means, log_stds


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
        self.up_conv_a = wn(nn.Conv2d(n_in, n_out, filter_size, stride, padding))
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
            input = F.upsample(input, scale_factor=0.5)

        return input + 0.1 * h
        

    def down(self, input, sample=False, return_z=False):
        x = F.elu(input)
        x = self.down_conv_a(x)
        
        pz_mean, pz_logsd, rz_mean, rz_logsd, down_context, h_det = x.split([self.z_size] * 4 + [self.h_size] * 2, 1)
        prior = D.Normal(pz_mean, torch.exp(pz_logsd))
            
        if sample:
            z = prior.rsample()
            kl = kl_obj = torch.zeros(input.size(0)).to(input.device)
            logqs = None
        else:
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
            input = F.upsample(input, scale_factor=2.)
        
        h = self.down_conv_b(h)

        if return_z:
            return input + 0.1 * h, kl, kl_obj, z, logqs
        else:
            return input + 0.1 * h, kl, kl_obj 

