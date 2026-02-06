#!/usr/bin/env python3
"""
Run 4-way experiment with PARAMETER-MATCHED configurations:
1. Diagonal Gaussian posterior + standard ELBO (no IAF, matched params)
2. IAF posterior + standard ELBO (with IAF, base params)
3. Diagonal Gaussian posterior + free bits (no IAF, matched params)
4. IAF posterior + free bits (with IAF, base params)

The models WITHOUT IAF use stronger decoder to match parameter count of models WITH IAF.
"""

import subprocess
import sys
import os
import torch
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
                    if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                        iaf_params += sum(p.numel() for p in module.parameters())
    return iaf_params

def find_matching_config(base_config, target_params, tolerance=0.02):
    """
    Find configuration without IAF that matches target parameter count.
    """
    print(f"  Target: {target_params:,} parameters")
    print(f"  Tolerance: {tolerance*100:.1f}%")
    
    best_config = None
    best_diff = float('inf')
    best_params = 0
    
    # Try different h_size and n_blocks combinations
    test_config = base_config.copy()
    test_config['iaf'] = 0
    
    # Strategy 1: Increase h_size
    print("\n  Trying h_size adjustments...")
    for h_mult in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0, 2.2, 2.5]:
        test_config['h_size'] = int(base_config['h_size'] * h_mult)
        args = argparse.Namespace(**test_config)
        try:
            model = VAE(args)
            params = count_parameters(model)
            diff = abs(params - target_params) / target_params
            
            if diff < best_diff:
                best_diff = diff
                best_config = test_config.copy()
                best_params = params
                
            if diff <= tolerance:
                print(f"    ✓ Found match: h_size={test_config['h_size']}, params={params:,}, diff={diff*100:.2f}%")
                return test_config, params
        except Exception as e:
            continue
    
    # Strategy 2: Increase n_blocks
    print("\n  Trying n_blocks adjustments...")
    for n_add in [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16]:
        test_config = base_config.copy()
        test_config['iaf'] = 0
        test_config['n_blocks'] = base_config['n_blocks'] + n_add
        
        args = argparse.Namespace(**test_config)
        try:
            model = VAE(args)
            params = count_parameters(model)
            diff = abs(params - target_params) / target_params
            
            if diff < best_diff:
                best_diff = diff
                best_config = test_config.copy()
                best_params = params
                
            if diff <= tolerance:
                print(f"    ✓ Found match: n_blocks={test_config['n_blocks']}, params={params:,}, diff={diff*100:.2f}%")
                return test_config, params
        except Exception as e:
            continue
    
    # Strategy 3: Combinations
    print("\n  Trying combinations...")
    for h_mult in [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]:
        for n_add in [0, 1, 2, 3, 4, 5]:
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
                    best_params = params
                    
                if diff <= tolerance:
                    print(f"    ✓ Found match: h_size={test_config['h_size']}, n_blocks={test_config['n_blocks']}, params={params:,}, diff={diff*100:.2f}%")
                    return test_config, params
            except Exception as e:
                continue
    
    print(f"    Best match: h_size={best_config['h_size']}, n_blocks={best_config['n_blocks']}, params={best_params:,}, diff={best_diff*100:.2f}%")
    return best_config, best_params

# Base configuration WITH IAF (default encoder/decoder)
BASE_CONFIG_WITH_IAF = {
    'n_blocks': 4,      # Default
    'depth': 2,         # Default
    'h_size': 64,       # Default
    'z_size': 32,
    'iaf': 1,
}

# Training settings
TRAINING_CONFIG = {
    'batch_size': 128,
    'lr': 0.002,
    'n_epochs': 100,
    'free_bits_value': 0.1,
}

# Experiment configurations
EXPERIMENTS = [
    {
        'name': 'diag_gauss_standard_elbo',
        'description': 'Diagonal Gaussian posterior + standard ELBO (matched params)',
        'iaf': 0,
        'free_bits': 0.0,
        'use_matched': True,  # Use matched config (stronger decoder)
    },
    {
        'name': 'iaf_standard_elbo',
        'description': 'IAF posterior + standard ELBO (base params)',
        'iaf': 1,
        'free_bits': 0.0,
        'use_matched': False,  # Use base config
    },
    {
        'name': 'diag_gauss_free_bits',
        'description': 'Diagonal Gaussian posterior + free bits (matched params)',
        'iaf': 0,
        'free_bits': TRAINING_CONFIG['free_bits_value'],
        'use_matched': True,  # Use matched config (stronger decoder)
    },
    {
        'name': 'iaf_free_bits',
        'description': 'IAF posterior + free bits (base params)',
        'iaf': 1,
        'free_bits': TRAINING_CONFIG['free_bits_value'],
        'use_matched': False,  # Use base config
    },
]

def run_experiment(exp_config, arch_config):
    """Run a single experiment configuration."""
    print("\n" + "="*80)
    print(f"Running: {exp_config['description']}")
    print(f"Architecture: n_blocks={arch_config['n_blocks']}, h_size={arch_config['h_size']}, depth={arch_config['depth']}")
    print(f"Configuration: IAF={exp_config['iaf']}, free_bits={exp_config['free_bits']}")
    print("="*80 + "\n")
    
    cmd = [
        sys.executable, 'main.py',
        '--n_blocks', str(arch_config['n_blocks']),
        '--depth', str(arch_config['depth']),
        '--h_size', str(arch_config['h_size']),
        '--z_size', str(arch_config['z_size']),
        '--batch_size', str(TRAINING_CONFIG['batch_size']),
        '--lr', str(TRAINING_CONFIG['lr']),
        '--n_epochs', str(TRAINING_CONFIG['n_epochs']),
        '--free_bits', str(exp_config['free_bits']),
        '--iaf', str(exp_config['iaf']),
    ]
    
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    
    if result.returncode != 0:
        print(f"\n❌ Experiment '{exp_config['name']}' failed with return code {result.returncode}")
        return False
    else:
        print(f"\n✅ Experiment '{exp_config['name']}' completed successfully")
        return True

def main():
    print("="*80)
    print("4-Way IAF vs Free Bits Experiment (PARAMETER-MATCHED)")
    print("="*80)
    
    # Step 1: Count parameters in base model WITH IAF
    print("\nStep 1: Analyzing base model WITH IAF...")
    base_args = argparse.Namespace(**BASE_CONFIG_WITH_IAF)
    base_model = VAE(base_args)
    base_params = count_parameters(base_model)
    iaf_params = count_iaf_params(base_model)
    
    print(f"\nBase Configuration (WITH IAF):")
    for k, v in BASE_CONFIG_WITH_IAF.items():
        print(f"  {k}: {v}")
    print(f"\nParameter Count:")
    print(f"  Total: {base_params:,}")
    print(f"  IAF components: {iaf_params:,}")
    print(f"  Non-IAF: {base_params - iaf_params:,}")
    
    # Step 2: Find matching configuration WITHOUT IAF
    print("\n" + "="*80)
    print("Step 2: Finding parameter-matched configuration WITHOUT IAF...")
    print("="*80)
    
    matched_config, matched_params = find_matching_config(
        BASE_CONFIG_WITH_IAF, base_params, tolerance=0.02
    )
    
    print(f"\nMatched Configuration (WITHOUT IAF):")
    for k, v in matched_config.items():
        print(f"  {k}: {v}")
    print(f"\nParameter Count:")
    print(f"  Total: {matched_params:,}")
    print(f"  Difference: {abs(matched_params - base_params):,} ({abs(matched_params - base_params)/base_params*100:.2f}%)")
    
    # Step 3: Prepare experiment configurations
    print("\n" + "="*80)
    print("Step 3: Experiment Configurations")
    print("="*80)
    print("\nExperiments to run:")
    for i, exp in enumerate(EXPERIMENTS, 1):
        config = matched_config if exp['use_matched'] else BASE_CONFIG_WITH_IAF
        print(f"  {i}. {exp['description']}")
        print(f"     Config: n_blocks={config['n_blocks']}, h_size={config['h_size']}, IAF={exp['iaf']}, free_bits={exp['free_bits']}")
    
    # Step 4: Run experiments
    print("\n" + "="*80)
    print("Step 4: Running Experiments")
    print("="*80)
    
    results = []
    for i, exp_config in enumerate(EXPERIMENTS, 1):
        print(f"\n[{i}/4] Starting experiment: {exp_config['name']}")
        
        # Select architecture config
        arch_config = matched_config if exp_config['use_matched'] else BASE_CONFIG_WITH_IAF
        
        success = run_experiment(exp_config, arch_config)
        results.append((exp_config['name'], success))
        
        if not success:
            print(f"\n⚠️  Warning: Experiment {i} failed. Continuing with next experiment...")
    
    # Summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    for name, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{status}: {name}")
    
    successful = sum(1 for _, s in results if s)
    print(f"\nCompleted {successful}/{len(EXPERIMENTS)} experiments successfully")
    print("\nResults are logged in:")
    print("  - TensorBoard logs: runs/<model_name>/")
    print("  - Model checkpoints: runs/<model_name>/best_model.pth")
    print("\nTo view TensorBoard logs:")
    print("  tensorboard --logdir runs/")
    print("="*80)
    
    print("\nParameter Matching Summary:")
    print(f"  Base (WITH IAF): {base_params:,} params")
    print(f"  Matched (WITHOUT IAF): {matched_params:,} params")
    print(f"  Difference: {abs(matched_params - base_params)/base_params*100:.2f}%")

if __name__ == '__main__':
    main()
