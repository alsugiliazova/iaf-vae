#!/usr/bin/env python3
"""
Utility to count parameters in VAE models with and without IAF.
Helps match parameter counts between configurations.
"""

import torch
import torch.nn as nn
import argparse
from main import VAE

def count_parameters(model):
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_parameters_by_component(model, args):
    """Count parameters by component type."""
    counts = {
        'first_conv': 0,
        'last_conv': 0,
        'iaf_layers': 0,
        'iaf_ar_conv': 0,  # IAF autoregressive convolutions
        'other': 0,
    }
    
    # First and last conv
    counts['first_conv'] = sum(p.numel() for p in model.first_conv.parameters())
    counts['last_conv'] = sum(p.numel() for p in model.last_conv.parameters())
    
    # Count IAF layers
    for layer_group in model.layers:
        for layer in layer_group:
            # Count all conv layers in IAFLayer
            for name, module in layer.named_modules():
                if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                    param_count = sum(p.numel() for p in module.parameters())
                    if 'down_ar_conv' in name:
                        counts['iaf_ar_conv'] += param_count
                    else:
                        counts['iaf_layers'] += param_count
    
    # Other parameters (h, dec_log_stdv) - count directly
    counts['other'] = model.h.numel() + model.dec_log_stdv.numel()
    
    return counts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_blocks', type=int, default=4)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--z_size', type=int, default=32)
    parser.add_argument('--h_size', type=int, default=64)
    parser.add_argument('--iaf', type=int, default=1)
    args = parser.parse_args()
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VAE(args).to(device)
    
    total_params = count_parameters(model)
    component_params = count_parameters_by_component(model, args)
    
    print("="*60)
    print("Parameter Count Analysis")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  n_blocks: {args.n_blocks}")
    print(f"  depth: {args.depth}")
    print(f"  h_size: {args.h_size}")
    print(f"  z_size: {args.z_size}")
    print(f"  IAF: {args.iaf}")
    
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"\nBy Component:")
    print(f"  First conv: {component_params['first_conv']:,}")
    print(f"  Last conv: {component_params['last_conv']:,}")
    print(f"  IAF layers (up/down convs): {component_params['iaf_layers']:,}")
    if args.iaf:
        print(f"  IAF AR conv (autoregressive): {component_params['iaf_ar_conv']:,}")
    print(f"  Other (h, dec_log_stdv): {component_params['other']:,}")
    
    if args.iaf:
        print(f"\nIAF AR conv params: {component_params['iaf_ar_conv']:,}")
        print(f"Non-IAF params (if IAF removed): {total_params - component_params['iaf_ar_conv']:,}")
    
    print("="*60)

if __name__ == '__main__':
    main()
