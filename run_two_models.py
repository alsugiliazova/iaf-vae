#!/usr/bin/env python3
"""
Run 2 models with matched parameters:

1. Matched VAE: Strong decoder + free bits (no IAF, same parameter count)
2. IAF model (from paper): Default encoder/decoder + IAF

This compares whether IAF improves reconstruction when latent usage is enforced.
"""

import subprocess
import sys
import os
import torch
import argparse
import time
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

# Base configuration WITH IAF (from paper - default encoder/decoder)
BASE_CONFIG_WITH_IAF = {
    'n_blocks': 4,      # Default from paper
    'depth': 2,         # Default from paper
    'h_size': 64,       # Default from paper
    'z_size': 32,
    'iaf': 1,
}

# Training settings
TRAINING_CONFIG = {
    'batch_size': 128,
    'lr': 0.002,
    'n_epochs': 100,
    'free_bits_value': 0.1,  # Free bits for matched model
}

# Two models to compare (non-IAF first, then IAF)
MODELS = [
    {
        'name': 'matched_vae',
        'description': 'Matched VAE: Strong decoder + free bits (no IAF)',
        'iaf': 0,
        'free_bits': TRAINING_CONFIG['free_bits_value'],
        'use_base_config': False,  # Use matched config
    },
    {
        'name': 'iaf_model',
        'description': 'IAF model (from paper): Default encoder/decoder + IAF',
        'iaf': 1,
        'free_bits': 0.0,  # Standard ELBO
        'use_base_config': True,  # Use base config
    },
]

def run_experiment(model_config, arch_config, model_num, total_models):
    """Run a single model with timing."""
    print("\n" + "="*80)
    print(f"Running [{model_num}/{total_models}]: {model_config['description']}")
    print(f"Architecture: n_blocks={arch_config['n_blocks']}, h_size={arch_config['h_size']}, depth={arch_config['depth']}")
    print(f"Configuration: IAF={model_config['iaf']}, free_bits={model_config['free_bits']}")
    print("="*80 + "\n")
    
    # Count parameters for this model
    args = argparse.Namespace(**arch_config)
    args.iaf = model_config['iaf']
    model = VAE(args)
    model_params = count_parameters(model)
    print(f"Model Parameters: {model_params:,}")
    print()
    
    cmd = [
        sys.executable, 'main.py',
        '--n_blocks', str(arch_config['n_blocks']),
        '--depth', str(arch_config['depth']),
        '--h_size', str(arch_config['h_size']),
        '--z_size', str(arch_config['z_size']),
        '--batch_size', str(TRAINING_CONFIG['batch_size']),
        '--lr', str(TRAINING_CONFIG['lr']),
        '--n_epochs', str(TRAINING_CONFIG['n_epochs']),
        '--free_bits', str(model_config['free_bits']),
        '--iaf', str(model_config['iaf']),
    ]
    
    print(f"Command: {' '.join(cmd)}\n")
    
    # Track timing
    start_time = time.time()
    epoch_times = []
    
    # We need to modify main.py to output timing, or parse the output
    # For now, we'll track total time and estimate per epoch
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    
    total_time = time.time() - start_time
    avg_epoch_time = total_time / TRAINING_CONFIG['n_epochs']
    
    if result.returncode != 0:
        print(f"\n❌ Model '{model_config['name']}' failed with return code {result.returncode}")
        print(f"Total time: {total_time/60:.2f} minutes")
        return False, total_time, avg_epoch_time
    else:
        print(f"\n✅ Model '{model_config['name']}' completed successfully")
        print(f"Total time: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")
        print(f"Average per epoch: {avg_epoch_time:.2f} seconds ({avg_epoch_time/60:.2f} minutes)")
        return True, total_time, avg_epoch_time

def main():
    print("="*80)
    print("2-Model Comparison: IAF vs Matched VAE")
    print("="*80)
    
    # Step 1: Count parameters in base model WITH IAF
    print("\nStep 1: Analyzing IAF model (from paper)...")
    base_args = argparse.Namespace(**BASE_CONFIG_WITH_IAF)
    base_model = VAE(base_args)
    base_params = count_parameters(base_model)
    iaf_params = count_iaf_params(base_model)
    
    print(f"\nIAF Model Configuration:")
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
    
    print(f"\nMatched VAE Configuration:")
    for k, v in matched_config.items():
        print(f"  {k}: {v}")
    print(f"\nParameter Count:")
    print(f"  Total: {matched_params:,}")
    print(f"  Difference: {abs(matched_params - base_params):,} ({abs(matched_params - base_params)/base_params*100:.2f}%)")
    
    # Print parameter summary at the beginning
    print("\n" + "="*80)
    print("PARAMETER SUMMARY")
    print("="*80)
    print(f"Model 1 (Matched VAE - no IAF):")
    print(f"  Config: n_blocks={matched_config['n_blocks']}, h_size={matched_config['h_size']}, depth={matched_config['depth']}")
    print(f"  Parameters: {matched_params:,}")
    print(f"\nModel 2 (IAF Model):")
    print(f"  Config: n_blocks={BASE_CONFIG_WITH_IAF['n_blocks']}, h_size={BASE_CONFIG_WITH_IAF['h_size']}, depth={BASE_CONFIG_WITH_IAF['depth']}")
    print(f"  Parameters: {base_params:,}")
    print(f"  Difference: {abs(matched_params - base_params):,} ({abs(matched_params - base_params)/base_params*100:.2f}%)")
    print("="*80)
    
    # Step 3: Prepare model configurations
    print("\n" + "="*80)
    print("Step 3: Model Configurations")
    print("="*80)
    print("\nModels to train:")
    for i, model in enumerate(MODELS, 1):
        config = BASE_CONFIG_WITH_IAF if model['use_base_config'] else matched_config
        print(f"  {i}. {model['description']}")
        print(f"     Config: n_blocks={config['n_blocks']}, h_size={config['h_size']}, IAF={model['iaf']}, free_bits={model['free_bits']}")
        print(f"     Parameters: {base_params if model['use_base_config'] else matched_params:,}")
    
    # Step 4: Run experiments
    print("\n" + "="*80)
    print("Step 4: Training Models")
    print("="*80)
    
    overall_start_time = time.time()
    results = []
    timing_info = []
    
    for i, model_config in enumerate(MODELS, 1):
        print(f"\n[{i}/{len(MODELS)}] Starting: {model_config['name']}")
        
        # Select architecture config
        arch_config = BASE_CONFIG_WITH_IAF if model_config['use_base_config'] else matched_config
        
        success, total_time, avg_epoch_time = run_experiment(model_config, arch_config, i, len(MODELS))
        results.append((model_config['name'], success))
        timing_info.append({
            'name': model_config['name'],
            'total_time': total_time,
            'avg_epoch_time': avg_epoch_time,
        })
        
        if not success:
            print(f"\n⚠️  Warning: Model {i} failed. Continuing with next model...")
    
    overall_total_time = time.time() - overall_start_time
    
    # Summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    for name, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{status}: {name}")
    
    successful = sum(1 for _, s in results if s)
    print(f"\nCompleted {successful}/{len(MODELS)} models successfully")
    
    print("\n" + "="*80)
    print("TIMING SUMMARY")
    print("="*80)
    for timing in timing_info:
        print(f"\n{timing['name']}:")
        print(f"  Total time: {timing['total_time']/60:.2f} minutes ({timing['total_time']/3600:.2f} hours)")
        print(f"  Average per epoch: {timing['avg_epoch_time']:.2f} seconds ({timing['avg_epoch_time']/60:.2f} minutes)")
    
    print(f"\nOverall total time: {overall_total_time/60:.2f} minutes ({overall_total_time/3600:.2f} hours)")
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print("\nResults are logged in:")
    print("  - TensorBoard logs: runs/<model_name>/")
    print("  - Model checkpoints: runs/<model_name>/best_model.pth")
    print("\nTo view TensorBoard logs:")
    print("  tensorboard --logdir runs/")
    print("="*80)
    
    print("\nParameter Matching Summary:")
    print(f"  Model 1 (Matched VAE): {matched_params:,} params")
    print(f"  Model 2 (IAF Model): {base_params:,} params")
    print(f"  Difference: {abs(matched_params - base_params):,} ({abs(matched_params - base_params)/base_params*100:.2f}%)")
    print("\nComparison:")
    print("  Both models have approximately the same number of parameters.")
    print("  Compare reconstruction quality, KL divergence, and ELBO in TensorBoard.")

if __name__ == '__main__':
    main()
