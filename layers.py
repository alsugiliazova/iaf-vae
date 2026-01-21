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
            # Small initialization - mask is applied automatically in forward pass
            nn.init.normal_(input_layer.weight, mean=0.0, std=0.01)
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
                nn.init.normal_(hidden_layer.weight, mean=0.0, std=0.01)
                hidden_layer.bias.data.zero_()
            layers.append(hidden_layer)
            layers.append(nn.ELU())
        
        # Output layer: hidden_dim -> 2*z_dim (mean and log_std for each dimension)
        mask_out = masks[-1]
        if mask_out.shape != (2 * z_dim, hidden_dim):
            raise ValueError(f"Output mask shape {mask_out.shape} doesn't match expected ({2 * z_dim}, {hidden_dim})")
        output_layer = MaskedLinear(hidden_dim, 2 * z_dim, mask_out)
        
        # Initialize output layer to produce reasonable priors
        # CRITICAL: Initialize close to standard Normal N(0,1) to match baseline
        with torch.no_grad():
            # Means: initialize to exactly zero (prior mean = 0)
            # Mask is applied automatically in forward pass
            output_layer.weight.data[:z_dim, :].zero_()
            output_layer.bias.data[:z_dim].zero_()
            # Log_stds: initialize to zero (std = 1, matching standard Normal)
            output_layer.weight.data[z_dim:, :].zero_()
            output_layer.bias.data[z_dim:].zero_()  # log_std = 0 means std = 1
        
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
        
        # CRITICAL: Constrain log_std to reasonable range
        # This prevents the prior from becoming too sharp (which causes huge KL)
        # Clamp to ensure std is between ~0.1 and ~2.7 (reasonable for Normal priors)
        log_stds = torch.clamp(log_stds, min=-2.3, max=1.0)
        
        # Compute standard deviations
        stds = torch.exp(log_stds)
        
        # Additional safety: ensure minimum std
        stds = torch.clamp(stds, min=0.1, max=2.7)
        
        # Compute log probability: log p(z) = Σ_i log N(z_i | μ_i, σ_i)
        # PyTorch's Normal.log_prob already includes the normalization term
        dist = D.Normal(means, stds)
        log_prob = dist.log_prob(z_flat).sum(dim=1)  # Sum over dimensions
        
        return log_prob, means, log_stds


# Convolutional Autoregressive Prior (PixelCNN-style)
# -------------------------------------------------------------------------------------------------------

class ConvARPrior(nn.Module):
    """
    Convolutional autoregressive prior p(z) using masked convolutions.
    
    Similar to PixelCNN, uses spatial autoregressive structure where
    z at position (h, w) depends on positions before it in raster order.
    
    This preserves spatial structure unlike the fully-connected MADE approach.
    """
    def __init__(self, z_size, h_size=64, n_layers=2):
        """
        Args:
            z_size: Number of channels in z (e.g., 32)
            h_size: Hidden channels for masked convolutions
            n_layers: Number of masked conv layers
        """
        super(ConvARPrior, self).__init__()
        self.z_size = z_size
        self.h_size = h_size
        
        # Build masked conv network
        # First layer: type A mask (excludes current pixel entirely)
        # Subsequent layers: type B mask (includes current pixel from previous layer)
        layers = []
        
        # Input layer (type A - strict autoregressive)
        layers.append(MaskedConv2d('A', z_size, h_size, 3, 1, 1))
        layers.append(nn.ELU())
        
        # Hidden layers (type B)
        for _ in range(n_layers - 1):
            layers.append(MaskedConv2d('B', h_size, h_size, 3, 1, 1))
            layers.append(nn.ELU())
        
        self.net = nn.Sequential(*layers)
        
        # Output layers for mean and log_std (type B)
        self.mean_conv = MaskedConv2d('B', h_size, z_size, 3, 1, 1)
        self.logstd_conv = MaskedConv2d('B', h_size, z_size, 3, 1, 1)
        
        # Initialize output layers to produce N(0,1) prior initially
        with torch.no_grad():
            self.mean_conv.weight.data.zero_()
            self.mean_conv.bias.data.zero_()
            self.logstd_conv.weight.data.zero_()
            self.logstd_conv.bias.data.zero_()  # log_std=0 means std=1
    
    def forward(self, z):
        """
        Compute autoregressive prior parameters for z.
        
        Args:
            z: Latent tensor of shape (B, z_size, H, W)
        
        Returns:
            means: Prior means (B, z_size, H, W)
            log_stds: Prior log standard deviations (B, z_size, H, W)
            log_probs: Log probability per dimension (B, z_size, H, W)
        """
        # Forward through masked conv network
        h = self.net(z)
        
        # Compute mean and log_std
        means = self.mean_conv(h)
        log_stds = self.logstd_conv(h)
        
        # Clamp log_std to reasonable range
        log_stds = torch.clamp(log_stds, min=-2.0, max=2.0)
        stds = torch.exp(log_stds)
        
        # Compute log probability per dimension
        dist = D.Normal(means, stds)
        log_probs = dist.log_prob(z)  # (B, z_size, H, W)
        
        return means, log_stds, log_probs


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
        
        # For IAF utilization analysis - stores stats from last forward pass
        self.iaf_stats = None

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
        
        # Reset IAF stats
        self.iaf_stats = None
            
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
                z_before = z  # Store z before transform for stats
                z = (z - arw_mean) / torch.exp(arw_logsd)
                
                # Store IAF statistics for analysis
                # These measure how "active" the IAF layer is
                with torch.no_grad():
                    # Compute norms by flattening spatial dims
                    z_diff = (z - z_before).view(z.size(0), -1)
                    z_before_flat = z_before.view(z.size(0), -1)
                    z_change_norm = z_diff.norm(dim=1).mean().item()
                    z_before_norm = z_before_flat.norm(dim=1).mean().item()
                    
                    self.iaf_stats = {
                        # Scale statistics (arw_logsd): if |s| ≈ 0, layer is near-identity
                        'scale_mean': arw_logsd.mean().item(),
                        'scale_std': arw_logsd.std().item(),
                        'scale_abs_mean': arw_logsd.abs().mean().item(),
                        'scale_max': arw_logsd.abs().max().item(),
                        # Shift statistics (arw_mean)
                        'shift_abs_mean': arw_mean.abs().mean().item(),
                        'shift_std': arw_mean.std().item(),
                        # Log-det of Jacobian (sum of log-scales)
                        'log_det_mean': arw_logsd.sum(dim=(1,2,3)).mean().item(),
                        # Change in z
                        'z_change_norm': z_change_norm,
                        'z_change_relative': z_change_norm / (z_before_norm + 1e-8),
                    }
            
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

