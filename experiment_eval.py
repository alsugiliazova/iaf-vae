"""
Evaluation script for encoder-decoder capacity matching experiment.

Evaluates matched and mismatched encoder-decoder pairs.
"""
import torch
import numpy as np
from torchvision import datasets, transforms
from modular_vae import ModularVAE, create_mismatched_vae
from utils import logistic_ll, reset_log, print_and_log_scalar
from collections import defaultdict as DD


def evaluate_model(model, test_loader, device='cuda'):
    """Evaluate model on test set"""
    model.eval()
    log = reset_log()
    
    with torch.no_grad():
        for batch_idx, (input, _) in enumerate(test_loader):
            input = input.cuda() if device == 'cuda' else input
            
            x, kl, kl_obj = model(input)
            
            log_pxz = logistic_ll(x, model.dec_log_stdv, sample=input)
            elbo = (kl - log_pxz)
            bpd = elbo / (32 * 32 * 3 * np.log(2.))
            
            log['kl'] += [kl.mean().item()]
            log['bpd'] += [bpd.mean().item()]
            log['elbo'] += [elbo.mean().item()]
            log['kl obj'] += [kl_obj.mean().item()]
            log['log p(x|z)'] += [log_pxz.mean().item()]
    
    # Average metrics
    metrics = {}
    for key, value in log.items():
        metrics[key] = sum(value) / len(value)
    
    return metrics


def run_experiment(weak_model_path, strong_model_path, test_loader, device='cuda', save_dir='./experiment_models'):
    """
    Run the encoder-decoder capacity matching experiment.
    
    Args:
        weak_model_path: Path to trained weak decoder model
        strong_model_path: Path to trained strong decoder model
        test_loader: Test data loader
        device: Device to run on
        save_dir: Directory to save all 4 models
    """
    import os
    
    print("=" * 80)
    print("Encoder-Decoder Capacity Matching Experiment")
    print("=" * 80)
    
    # Load models
    print("\nLoading models...")
    # Set weights_only=False for full model loading (PyTorch 2.6+ default changed)
    weak_model = torch.load(weak_model_path, map_location=device, weights_only=False)
    strong_model = torch.load(strong_model_path, map_location=device, weights_only=False)
    
    # Print architectures
    print("\n" + "=" * 80)
    print("Weak Decoder Model Architecture:")
    print("=" * 80)
    print(weak_model)
    
    print("\n" + "=" * 80)
    print("Strong Decoder Model Architecture:")
    print("=" * 80)
    print(strong_model)
    
    # Create 4 configurations
    print("\n" + "=" * 80)
    print("Creating configurations...")
    print("=" * 80)
    
    # 1. Matched Weak: Encoder_Weak + Decoder_Weak
    matched_weak = weak_model
    print("\n1. Matched Weak (Encoder_Weak + Decoder_Weak)")
    print(matched_weak)
    
    # 2. Matched Strong: Encoder_Strong + Decoder_Strong
    matched_strong = strong_model
    print("\n2. Matched Strong (Encoder_Strong + Decoder_Strong)")
    print(matched_strong)
    
    # 3. Mismatched 1: Encoder_Weak + Decoder_Strong
    mismatched_weak_enc_strong_dec = create_mismatched_vae(weak_model, strong_model)
    mismatched_weak_enc_strong_dec = mismatched_weak_enc_strong_dec.to(device)
    print("\n3. Mismatched 1 (Encoder_Weak + Decoder_Strong)")
    print(mismatched_weak_enc_strong_dec)
    
    # 4. Mismatched 2: Encoder_Strong + Decoder_Weak
    mismatched_strong_enc_weak_dec = create_mismatched_vae(strong_model, weak_model)
    mismatched_strong_enc_weak_dec = mismatched_strong_enc_weak_dec.to(device)
    print("\n4. Mismatched 2 (Encoder_Strong + Decoder_Weak)")
    print(mismatched_strong_enc_weak_dec)
    
    # Save all 4 models
    print("\n" + "=" * 80)
    print("Saving all 4 models...")
    print("=" * 80)
    os.makedirs(save_dir, exist_ok=True)
    
    torch.save(matched_weak, os.path.join(save_dir, 'matched_weak.pth'))
    print(f"  Saved: {os.path.join(save_dir, 'matched_weak.pth')}")
    
    torch.save(matched_strong, os.path.join(save_dir, 'matched_strong.pth'))
    print(f"  Saved: {os.path.join(save_dir, 'matched_strong.pth')}")
    
    torch.save(mismatched_weak_enc_strong_dec, os.path.join(save_dir, 'mismatched_weak_enc_strong_dec.pth'))
    print(f"  Saved: {os.path.join(save_dir, 'mismatched_weak_enc_strong_dec.pth')}")
    
    torch.save(mismatched_strong_enc_weak_dec, os.path.join(save_dir, 'mismatched_strong_enc_weak_dec.pth'))
    print(f"  Saved: {os.path.join(save_dir, 'mismatched_strong_enc_weak_dec.pth')}")
    
    # Evaluate all configurations
    print("\n" + "=" * 80)
    print("Evaluating configurations...")
    print("=" * 80)
    results = {}
    
    print("\n1. Matched Weak (Encoder_Weak + Decoder_Weak)")
    results['matched_weak'] = evaluate_model(matched_weak, test_loader, device)
    print_metrics(results['matched_weak'])
    
    print("\n2. Matched Strong (Encoder_Strong + Decoder_Strong)")
    results['matched_strong'] = evaluate_model(matched_strong, test_loader, device)
    print_metrics(results['matched_strong'])
    
    print("\n3. Mismatched 1 (Encoder_Weak + Decoder_Strong)")
    results['mismatched_weak_enc'] = evaluate_model(mismatched_weak_enc_strong_dec, test_loader, device)
    print_metrics(results['mismatched_weak_enc'])
    
    print("\n4. Mismatched 2 (Encoder_Strong + Decoder_Weak)")
    results['mismatched_strong_enc'] = evaluate_model(mismatched_strong_enc_weak_dec, test_loader, device)
    print_metrics(results['mismatched_strong_enc'])
    
    # Compare results
    print("\n" + "=" * 80)
    print("Comparison Summary")
    print("=" * 80)
    print_comparison(results)
    
    return results


def print_metrics(metrics):
    """Print metrics nicely"""
    print(f"  BPD: {metrics['bpd']:.4f}")
    print(f"  ELBO: {metrics['elbo']:.4f}")
    print(f"  KL: {metrics['kl']:.4f}")
    print(f"  log p(x|z): {metrics['log p(x|z)']:.4f}")


def print_comparison(results):
    """Print comparison table"""
    print("\nBits Per Dimension (BPD):")
    print(f"  Matched Weak:        {results['matched_weak']['bpd']:.4f}")
    print(f"  Matched Strong:       {results['matched_strong']['bpd']:.4f}")
    print(f"  Weak Enc + Strong Dec: {results['mismatched_weak_enc']['bpd']:.4f}")
    print(f"  Strong Enc + Weak Dec: {results['mismatched_strong_enc']['bpd']:.4f}")
    
    print("\nELBO:")
    print(f"  Matched Weak:        {results['matched_weak']['elbo']:.4f}")
    print(f"  Matched Strong:       {results['matched_strong']['elbo']:.4f}")
    print(f"  Weak Enc + Strong Dec: {results['mismatched_weak_enc']['elbo']:.4f}")
    print(f"  Strong Enc + Weak Dec: {results['mismatched_strong_enc']['elbo']:.4f}")
    
    print("\nKL Divergence:")
    print(f"  Matched Weak:        {results['matched_weak']['kl']:.4f}")
    print(f"  Matched Strong:       {results['matched_strong']['kl']:.4f}")
    print(f"  Weak Enc + Strong Dec: {results['mismatched_weak_enc']['kl']:.4f}")
    print(f"  Strong Enc + Weak Dec: {results['mismatched_strong_enc']['kl']:.4f}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--weak_model', type=str, required=True, help='Path to weak decoder model')
    parser.add_argument('--strong_model', type=str, required=True, help='Path to strong decoder model')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory for dataset')
    parser.add_argument('--save_dir', type=str, default='./experiment_models', help='Directory to save all 4 models')
    args = parser.parse_args()
    
    # Create test loader
    ds_transforms = transforms.Compose([transforms.ToTensor(), lambda x: x - 0.5])
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(args.data_dir, train=False, download=True, transform=ds_transforms),
        batch_size=args.batch_size, shuffle=False
    )
    
    # Run experiment
    results = run_experiment(args.weak_model, args.strong_model, test_loader, args.device, args.save_dir)
