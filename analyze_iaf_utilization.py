"""
Analyze IAF Layer Utilization

This script measures whether IAF posteriors actually use their flexibility.
For each IAF layer, it tracks:
- Scale factor statistics (if |s| ≈ 0, layer is near-identity)
- Shift factor statistics
- Log-determinant of Jacobian
- Change in z after the IAF transform

Usage:
    python analyze_iaf_utilization.py --model_path runs/YOUR_MODEL/best_model.pth

Or for quick test with current training:
    python analyze_iaf_utilization.py --model_path runs/NB4_D2_Z32_H64_BS128_FB0.2_LR0.002_IAF1_AR0/best_model.pth
"""

import torch
import argparse
import numpy as np
from collections import defaultdict
from torchvision import datasets, transforms

# Import model definition
from main import VAE


def collect_iaf_stats(model, dataloader, n_batches=50):
    """
    Run inference on data and collect IAF statistics from each layer.
    
    Returns:
        dict: Statistics per layer, keyed by (depth_idx, block_idx)
    """
    model.eval()
    
    # Collect stats across batches
    layer_stats = defaultdict(lambda: defaultdict(list))
    
    with torch.no_grad():
        for batch_idx, (input, _) in enumerate(dataloader):
            if batch_idx >= n_batches:
                break
            
            input = input.cuda()
            
            # Forward pass
            _ = model(input)
            
            # Collect stats from each IAFLayer
            for depth_idx, layer in enumerate(reversed(model.layers)):
                for block_idx, sub_layer in enumerate(reversed(layer)):
                    if sub_layer.iaf and sub_layer.iaf_stats is not None:
                        key = (depth_idx, block_idx)
                        for stat_name, stat_value in sub_layer.iaf_stats.items():
                            layer_stats[key][stat_name].append(stat_value)
    
    # Average across batches
    averaged_stats = {}
    for key, stats in layer_stats.items():
        averaged_stats[key] = {
            stat_name: {
                'mean': np.mean(values),
                'std': np.std(values),
            }
            for stat_name, values in stats.items()
        }
    
    return averaged_stats


def print_stats_table(stats):
    """Print a formatted table of IAF utilization statistics."""
    print("\n" + "="*80)
    print("IAF LAYER UTILIZATION ANALYSIS")
    print("="*80)
    print("\nKey metrics:")
    print("  - scale_abs_mean: Mean |s| of scale factor. If ≈0, layer is near-identity.")
    print("  - z_change_relative: Relative change in z. Higher = more transformation.")
    print("  - log_det_mean: Log-determinant of Jacobian. Measures volume change.")
    print()
    
    # Sort by layer order
    sorted_keys = sorted(stats.keys())
    
    # Print header
    print(f"{'Layer':<12} | {'|Scale|':<12} | {'|Shift|':<12} | {'z Change %':<12} | {'Log-Det':<12}")
    print("-"*70)
    
    for key in sorted_keys:
        depth_idx, block_idx = key
        layer_name = f"D{depth_idx}_B{block_idx}"
        s = stats[key]
        
        scale_abs = s['scale_abs_mean']['mean']
        shift_abs = s['shift_abs_mean']['mean']
        z_change = s['z_change_relative']['mean'] * 100  # Convert to percentage
        log_det = s['log_det_mean']['mean']
        
        print(f"{layer_name:<12} | {scale_abs:<12.4f} | {shift_abs:<12.4f} | {z_change:<12.2f} | {log_det:<12.2f}")
    
    print("-"*70)
    
    # Summary statistics
    all_scale_abs = [stats[k]['scale_abs_mean']['mean'] for k in sorted_keys]
    all_z_change = [stats[k]['z_change_relative']['mean'] * 100 for k in sorted_keys]
    
    print(f"\nSUMMARY:")
    print(f"  Average |scale| across layers: {np.mean(all_scale_abs):.4f}")
    print(f"  Min |scale|: {np.min(all_scale_abs):.4f} (Layer {sorted_keys[np.argmin(all_scale_abs)]})")
    print(f"  Max |scale|: {np.max(all_scale_abs):.4f} (Layer {sorted_keys[np.argmax(all_scale_abs)]})")
    print(f"  Average z change: {np.mean(all_z_change):.2f}%")
    
    # Interpretation
    print("\n" + "="*80)
    print("INTERPRETATION:")
    near_identity_threshold = 0.05
    near_identity_layers = [k for k in sorted_keys if stats[k]['scale_abs_mean']['mean'] < near_identity_threshold]
    active_layers = [k for k in sorted_keys if stats[k]['scale_abs_mean']['mean'] >= near_identity_threshold]
    
    if near_identity_layers:
        print(f"  Near-identity layers (|scale| < {near_identity_threshold}): {near_identity_layers}")
        print(f"  These layers contribute little transformation and may be unnecessary.")
    else:
        print(f"  All layers appear active (|scale| >= {near_identity_threshold})")
    
    if active_layers:
        print(f"  Active layers: {active_layers}")
    
    print("="*80 + "\n")


def save_stats_to_file(stats, output_path):
    """Save detailed statistics to a file."""
    sorted_keys = sorted(stats.keys())
    
    with open(output_path, 'w') as f:
        f.write("IAF Layer Utilization Statistics\n")
        f.write("="*60 + "\n\n")
        
        for key in sorted_keys:
            depth_idx, block_idx = key
            f.write(f"Layer D{depth_idx}_B{block_idx}:\n")
            
            for stat_name, values in stats[key].items():
                f.write(f"  {stat_name}: mean={values['mean']:.6f}, std={values['std']:.6f}\n")
            f.write("\n")
    
    print(f"Detailed stats saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze IAF layer utilization')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to trained model checkpoint (best_model.pth)')
    parser.add_argument('--n_batches', type=int, default=50,
                        help='Number of batches to analyze')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for analysis')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for detailed stats')
    
    # Model architecture args (must match trained model)
    parser.add_argument('--n_blocks', type=int, default=4)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--z_size', type=int, default=32)
    parser.add_argument('--h_size', type=int, default=64)
    parser.add_argument('--free_bits', type=float, default=0.2)
    parser.add_argument('--iaf', type=int, default=1)
    parser.add_argument('--ar_prior', type=int, default=0)
    
    args = parser.parse_args()
    
    print(f"Loading model from: {args.model_path}")
    
    # Create model with same architecture
    model = VAE(args).cuda()
    
    # Load trained weights
    state_dict = torch.load(args.model_path)
    model.load_state_dict(state_dict)
    print("Model loaded successfully!")
    
    # Check that IAF is enabled
    if not args.iaf:
        print("ERROR: This model does not have IAF enabled. Nothing to analyze.")
        exit(1)
    
    # Create test dataloader
    ds_transforms = transforms.Compose([transforms.ToTensor(), lambda x: x - 0.5])
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../cl-pytorch/data', train=False, download=True, transform=ds_transforms),
        batch_size=args.batch_size, shuffle=False, num_workers=1
    )
    
    print(f"Analyzing {args.n_batches} batches...")
    
    # Collect statistics
    stats = collect_iaf_stats(model, test_loader, n_batches=args.n_batches)
    
    # Print summary table
    print_stats_table(stats)
    
    # Optionally save detailed stats
    if args.output:
        save_stats_to_file(stats, args.output)
    else:
        # Default output path
        import os
        model_dir = os.path.dirname(args.model_path)
        default_output = os.path.join(model_dir, 'iaf_utilization_stats.txt')
        save_stats_to_file(stats, default_output)
