"""
Modular VAE for encoder-decoder capacity matching experiment.

This allows swapping decoders while keeping the same encoder/posterior.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as wn
import torch.distributions as D

from layers import IAFLayer


class WeakDecoder(nn.Module):
    """Weak decoder: MLP or small conv decoder"""
    def __init__(self, args, decoder_type='mlp'):
        super(WeakDecoder, self).__init__()
        self.args = args
        self.decoder_type = decoder_type
        self.hidden_dim = 512  # Hidden dimension for MLP
        
        if decoder_type == 'mlp':
            # MLP decoder: flatten → MLP → reshape
            # For CIFAR-10: after first_conv (4x4, stride=2, padding=1) -> 16x16
            # IAFLayer blocks maintain spatial size (stride=1, padding=1)
            # After down pass: h_det has shape (B, h_size, 16, 16)
            # So input_size = h_size * 16 * 16 = 160 * 16 * 16 = 40960
            # We'll create layers dynamically based on actual input size
            self.fc1 = None
            self.fc2 = None
            self.input_size = None
            
        elif decoder_type == 'small_conv':
            # Small conv decoder: 1-2 conv layers
            self.conv1 = wn(nn.Conv2d(args.h_size, args.h_size, 3, 1, 1))
            self.conv2 = wn(nn.Conv2d(args.h_size, args.h_size, 3, 1, 1))
    
    def forward(self, h):
        """
        Args:
            h: Hidden representation from encoder (B, h_size, H, W)
        Returns:
            x: Decoded representation (B, h_size, H, W)
        """
        if self.decoder_type == 'mlp':
            B, C, H, W = h.size()
            # Ensure we preserve the input channel dimension
            expected_C = self.args.h_size
            if C != expected_C:
                raise ValueError(f"MLP decoder expected input with {expected_C} channels, got {C}")
            
            input_size = C * H * W
            
            # Create layers dynamically on first forward or if size changed
            if self.fc1 is None or self.input_size != input_size:
                self.input_size = input_size
                device = h.device
                # Store original shape for reshaping
                self.original_shape = (C, H, W)
                self.fc1 = nn.Linear(input_size, self.hidden_dim).to(device)
                self.fc2 = nn.Linear(self.hidden_dim, input_size).to(device)
                # Register as modules so they're part of the model and trainable
                self.add_module('fc1', self.fc1)
                self.add_module('fc2', self.fc2)
            
            h_flat = h.view(B, -1)
            x = F.elu(self.fc1(h_flat))
            x = self.fc2(x)
            # Use stored original shape to ensure correct reshaping - must match input
            C_orig, H_orig, W_orig = self.original_shape
            assert C_orig == expected_C, f"Stored shape has {C_orig} channels, expected {expected_C}"
            x = x.view(B, C_orig, H_orig, W_orig)
            assert x.size(1) == expected_C, f"Reshaped output has {x.size(1)} channels, expected {expected_C}"
            return F.elu(x)
        
        elif self.decoder_type == 'small_conv':
            x = F.elu(self.conv1(h))
            x = F.elu(self.conv2(x))
            return x


class ResNetBlock(nn.Module):
    """ResNet block for strong decoder"""
    def __init__(self, channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = wn(nn.Conv2d(channels, channels, 3, 1, 1))
        self.conv2 = wn(nn.Conv2d(channels, channels, 3, 1, 1))
    
    def forward(self, x):
        residual = x
        out = F.elu(self.conv1(x))
        out = self.conv2(out)
        return F.elu(out + residual)


class StrongDecoder(nn.Module):
    """Strong decoder: Deep ResNet decoder"""
    def __init__(self, args, n_blocks=4):
        super(StrongDecoder, self).__init__()
        self.args = args
        
        # Multiple ResNet blocks
        blocks = []
        for _ in range(n_blocks):
            blocks.append(ResNetBlock(args.h_size))
        self.blocks = nn.ModuleList(blocks)
        
        # Additional conv layers
        self.conv1 = wn(nn.Conv2d(args.h_size, args.h_size, 3, 1, 1))
        self.conv2 = wn(nn.Conv2d(args.h_size, args.h_size, 3, 1, 1))
    
    def forward(self, h):
        """
        Args:
            h: Hidden representation from encoder (B, h_size, H, W)
        Returns:
            x: Decoded representation (B, h_size, H, W)
        """
        x = h
        for block in self.blocks:
            x = block(x)
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        return x


class ModularVAE(nn.Module):
    """
    Modular VAE that allows swapping decoders.
    
    Encoder: IAFLayer blocks (up pass) - same for both models
    Decoder: Can be weak (MLP/small conv) or strong (ResNet)
    """
    def __init__(self, args, decoder_type='weak', decoder_subtype='mlp'):
        super(ModularVAE, self).__init__()
        self.args = args
        self.decoder_type = decoder_type
        
        # Shared encoder components
        self.register_parameter('h', torch.nn.Parameter(torch.zeros(args.h_size)))
        self.register_parameter('dec_log_stdv', torch.nn.Parameter(torch.Tensor([0.])))
        
        # Encoder: IAFLayer blocks (up pass)
        layers = []
        for i in range(args.depth):
            layer = []
            for j in range(args.n_blocks):
                downsample = (i > 0) and (j == 0)
                layer += [IAFLayer(args, downsample)]
            layers += [nn.ModuleList(layer)]
        self.encoder_layers = nn.ModuleList(layers)
        
        self.first_conv = nn.Conv2d(3, args.h_size, 4, 2, 1)
        
        # Decoder: Different based on decoder_type
        if decoder_type == 'weak':
            self.decoder = WeakDecoder(args, decoder_subtype)
        elif decoder_type == 'strong':
            self.decoder = StrongDecoder(args)
        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")
        
        # Final output layer
        self.last_conv = nn.ConvTranspose2d(args.h_size, 3, 4, 2, 1)
    
    def encode(self, input):
        """Encode input through encoder (up pass)"""
        x = self.first_conv(input)
        h = self.h.view(1, -1, 1, 1)
        
        for layer in self.encoder_layers:
            for sub_layer in layer:
                x = sub_layer.up(x)
        
        h = h.expand_as(x)
        self.hid_shape = x[0].size()
        return h, x
    
    def decode(self, h, sample=False):
        """
        Decode through decoder and down pass.
        
        The down pass through IAFLayer blocks computes the posterior and samples z.
        After getting z, we apply the custom decoder to reconstruct x.
        
        Args:
            h: Initial hidden state
            sample: Whether to sample (skip KL computation)
        """
        kl, kl_obj = 0., 0.
        
        # Down pass through IAFLayer blocks (computes posterior, samples z, computes KL)
        # This is the "decoder" part that we want to vary
        # But it also computes the posterior, so we need to keep it
        # Instead, we'll replace the processing AFTER getting z
        
        # Store intermediate representations for custom decoder
        z_list = []
        h_list = []
        
        for layer in reversed(self.encoder_layers):
            for sub_layer in reversed(layer):
                h, curr_kl, curr_kl_obj = sub_layer.down(h, sample=sample)
                kl += curr_kl
                kl_obj += curr_kl_obj
                
                # Extract z and h_det from h (h = [z, h_det] after down pass)
                z_size = self.args.z_size
                z = h[:, :z_size, :, :]
                h_det = h[:, z_size:, :, :]
                z_list.append(z)
                h_list.append(h_det)
        
        # Use the final h for custom decoder
        # After all down passes, h has shape (B, h_size, H, W)
        # Note: down_conv_b outputs h_size channels, so h already has the correct shape
        # (The z and h_det are concatenated inside down(), but down_conv_b reduces it back to h_size)
        
        # Verify h has correct shape
        if h.size(1) != self.args.h_size:
            raise ValueError(f"After down pass, h has {h.size(1)} channels, expected {self.args.h_size}")
        
        # Apply custom decoder directly to h (which already has h_size channels)
        h_decoded = self.decoder(h)
        
        # Verify decoder output shape matches input
        if h_decoded.size(1) != self.args.h_size:
            raise ValueError(f"Decoder output has {h_decoded.size(1)} channels, expected {self.args.h_size}. Input h had {h.size(1)} channels")
        
        # Final output
        x = F.elu(h_decoded)
        x = self.last_conv(x)
        x = x.clamp(min=-0.5 + 1. / 512., max=0.5 - 1. / 512.)
        
        return x, kl, kl_obj
    
    def forward(self, input):
        """Full forward pass"""
        h, _ = self.encode(input)
        x, kl, kl_obj = self.decode(h, sample=False)
        return x, kl, kl_obj
    
    def sample(self, n_samples=64):
        """Sample from prior"""
        h = self.h.view(1, -1, 1, 1)
        h = h.expand((n_samples, *self.hid_shape))
        
        x, _, _ = self.decode(h, sample=True)
        return x
    
    def get_encoder(self):
        """Extract encoder components for mismatching"""
        return {
            'first_conv': self.first_conv,
            'encoder_layers': self.encoder_layers,
            'h': self.h,
            'hid_shape': self.hid_shape
        }
    
    def set_decoder(self, decoder):
        """Set decoder for mismatching"""
        self.decoder = decoder


def create_mismatched_vae(encoder_model, decoder_model):
    """
    Create a VAE with encoder from one model and decoder from another.
    
    Args:
        encoder_model: VAE model to extract encoder from
        decoder_model: VAE model to extract decoder from
    
    Returns:
        New VAE with mismatched encoder-decoder
    """
    args = encoder_model.args
    
    # Determine decoder type from decoder_model
    decoder_type = decoder_model.decoder_type
    decoder_subtype = getattr(decoder_model.decoder, 'decoder_type', None) if decoder_type == 'weak' else None
    
    # Create new model with the decoder's type (not encoder's type)
    if decoder_type == 'weak' and decoder_subtype:
        mismatched = ModularVAE(args, decoder_type='weak', decoder_subtype=decoder_subtype)
    elif decoder_type == 'strong':
        mismatched = ModularVAE(args, decoder_type='strong')
    else:
        mismatched = ModularVAE(args, decoder_type='weak')
    
    # Copy encoder components from encoder_model
    mismatched.first_conv.load_state_dict(encoder_model.first_conv.state_dict())
    mismatched.encoder_layers.load_state_dict(encoder_model.encoder_layers.state_dict())
    mismatched.h.data = encoder_model.h.data.clone()
    mismatched.hid_shape = encoder_model.hid_shape
    
    # For WeakDecoder with MLP, we need to initialize the layers first
    # by doing a forward pass, then load the state_dict
    if decoder_type == 'weak' and decoder_subtype == 'mlp':
        # Initialize WeakDecoder by doing a dummy forward pass
        # This creates fc1 and fc2 layers
        dummy_input = torch.zeros(1, args.h_size, 16, 16).to(encoder_model.h.device)
        _ = mismatched.decoder(dummy_input)
    
    # Copy decoder from decoder_model (now types match)
    mismatched.decoder.load_state_dict(decoder_model.decoder.state_dict())
    mismatched.last_conv.load_state_dict(decoder_model.last_conv.state_dict())
    mismatched.dec_log_stdv.data = decoder_model.dec_log_stdv.data.clone()
    
    return mismatched
