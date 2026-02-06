#!/usr/bin/env python3
"""
Find parameter-matched configurations:
- Model WITH IAF: default encoder/decoder + IAF
- Model WITHOUT IAF: stronger decoder + free bits (same total params)
"""

import torch
import torch.nn as nn
import argparse
from main import VAE

def count_parameters(model):
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_iaf_params(model):
    """Count parameters in IAF autoregressive convolutions."""
    iaf_params = 0
    for layer_group in model.layers:
        for layer in layer_group:
            if hasattr(layer, 'down_ar_conv') and layer.down_ar_conv is not None:
                for module in layer.down_ar_conv.modules():
                    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                        iaf_params += sum(p.numel() for p in module.parameters())
    return iaf_params

def find_matching_config(base_config, target_params, tolerance=0.01):
    """
    Find configuration without IAF that matches target parameter count.
    
    Args:
        base_config: Base configuration dict
        target_params: Target parameter count
        tolerance: Relative tolerance (0.01 = 1%)
    
    Returns:
        Matching configuration dict
    """
    print(f"\nTarget parameters: {target_params:,}")
    print(f"Tolerance: {tolerance*100:.1f}%")
    
    # Try different h_size and n_blocks combinations
    best_config = None
    best_diff = float('inf')
    
    # Base config without IAF
    test_config = base_config.copy()
    test_config['iaf'] = 0
    
    # Try increasing h_size first (affects all layers)
    for h_size_mult in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0]:
        test_config['h_size'] = int(base_config['h_size'] * h_size_mult)
        
        # Create model and count params
        args = argparse.Namespace(**test_config)
        try:
            model = VAE(args)
            params = count_parameters(model)
            diff = abs(params - target_params) / target_params
            
            print(f"  h_size={test_config['h_size']:3d}, params={params:10,}, diff={diff*100:5.2f}%")
            
            if diff < best_diff:
                best_diff = diff
                best_config = test_config.copy()
                
            if diff <= tolerance:
                print(f"\n✓ Found match within tolerance!")
                return test_config, params
        except Exception as e:
            continue
    
    # Try increasing n_blocks
    for n_blocks_add in [0, 1, 2, 3, 4, 5, 6, 8, 10]:
        test_config = base_config.copy()
        test_config['iaf'] = 0
        test_config['n_blocks'] = base_config['n_blocks'] + n_blocks_add
        
        args = argparse.Namespace(**test_config)
        try:
            model = VAE(args)
            params = count_parameters(model)
            diff = abs(params - target_params) / target_params
            
            print(f"  n_blocks={test_config['n_blocks']:2d}, params={params:10,}, diff={diff*100:5.2f}%")
            
            if diff < best_diff:
                best_diff = diff
                best_config = test_config.copy()
                
            if diff <= tolerance:
                print(f"\n✓ Found match within tolerance!")
                return test_config, params
        except Exception as e:
            continue
    
    # Try combinations
    print("\nTrying combinations...")
    for h_mult in [1.2, 1.3, 1.4, 1.5]:
        for n_add in [0, 1, 2, 3, 4]:
            test_config = base_config.copy()
            test_config['iaf'] = 0
            test_config['h_size'] = int(base_config['h_size'] * h_mult)
            test_config['n_blocks'] = base_config['n_blocks'] + n_add
            
            args = argparse.Namespace(**test_config)
            try:
                model = VAE(args)
                params = count_parameters(model)
                diff = abs(params - target_params) / target_params
                
                if diff < best_diff:
                    best_diff = diff
                    best_config = test_config.copy()
                    
                if diff <= tolerance:
                    print(f"\n✓ Found match within tolerance!")
                    return test_config, params
            except Exception as e:
                continue
    
    print(f"\nBest match: diff={best_diff*100:.2f}%")
    args = argparse.Namespace(**best_config)
    model = VAE(args)
    best_params = count_parameters(model)
    return best_config, best_params

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_n_blocks', type=int, default=4, help='Base n_blocks for model WITH IAF')
    parser.add_argument('--base_depth', type=int, default=2, help='Base depth')
    parser.add_argument('--base_h_size', type=int, default=64, help='Base h_size for model WITH IAF')
    parser.add_argument('--z_size', type=int, default=32)
    parser.add_argument('--tolerance', type=float, default=0.01, help='Parameter matching tolerance (0.01 = 1%%)')
    args = parser.parse_args()
    
    # Base configuration WITH IAF (default encoder/decoder)
    base_config = {
        'n_blocks': args.base_n_blocks,
        'depth': args.base_depth,
        'h_size': args.base_h_size,
        'z_size': args.z_size,
        'iaf': 1,  # WITH IAF
    }
    
    print("="*70)
    print("Finding Parameter-Matched Configurations")
    print("="*70)
    print("\nBase Configuration (WITH IAF):")
    for k, v in base_config.items():
        print(f"  {k}: {v}")
    
    # Create base model WITH IAF
    base_args = argparse.Namespace(**base_config)
    base_model = VAE(base_args)
    base_params = count_parameters(base_model)
    iaf_params = count_iaf_params(base_model)
    
    print(f"\nBase Model (WITH IAF):")
    print(f"  Total parameters: {base_params:,}")
    print(f"  IAF parameters: {iaf_params:,}")
    print(f"  Non-IAF parameters: {base_params - iaf_params:,}")
    
    # Find matching configuration WITHOUT IAF
    print("\n" + "="*70)
    print("Searching for matching configuration WITHOUT IAF...")
    print("="*70)
    
    matching_config, matching_params = find_matching_config(
        base_config, base_params, tolerance=args.tolerance
    )
    
    print("\n" + "="*70)
    print("Results")
    print("="*70)
    print(f"\nBase (WITH IAF):")
    print(f"  Config: n_blocks={base_config['n_blocks']}, h_size={base_config['h_size']}, depth={base_config['depth']}")
    print(f"  Parameters: {base_params:,}")
    
    print(f"\nMatched (WITHOUT IAF):")
    print(f"  Config: n_blocks={matching_config['n_blocks']}, h_size={matching_config['h_size']}, depth={matching_config['depth']}")
    print(f"  Parameters: {matching_params:,}")
    print(f"  Difference: {abs(matching_params - base_params):,} ({abs(matching_params - base_params)/base_params*100:.2f}%)")
    
    print("\n" + "="*70)
    print("Recommended Experiment Configurations")
    print("="*70)
    print("\n1. Diagonal Gaussian + Standard ELBO (no IAF, no free bits):")
    print(f"   --n_blocks {matching_config['n_blocks']} --depth {matching_config['depth']} --h_size {matching_config['h_size']} --z_size {matching_config['z_size']} --iaf 0 --free_bits 0.0")
    
    print("\n2. IAF + Standard ELBO (with IAF, no free bits):")
    print(f"   --n_blocks {base_config['n_blocks']} --depth {base_config['depth']} --h_size {base_config['h_size']} --z_size {base_config['z_size']} --iaf 1 --free_bits 0.0")
    
    print("\n3. Diagonal Gaussian + Free Bits (no IAF, with free bits):")
    print(f"   --n_blocks {matching_config['n_blocks']} --depth {matching_config['depth']} --h_size {matching_config['h_size']} --z_size {matching_config['z_size']} --iaf 0 --free_bits 0.1")
    
    print("\n4. IAF + Free Bits (with IAF, with free bits):")
    print(f"   --n_blocks {base_config['n_blocks']} --depth {base_config['depth']} --h_size {base_config['h_size']} --z_size {base_config['z_size']} --iaf 1 --free_bits 0.1")
    
    print("="*70)

if __name__ == '__main__':
    main()
