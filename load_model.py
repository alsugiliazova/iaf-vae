"""
Utility script to load and inspect saved models from the experiment.

Usage:
    python load_model.py --model_path experiment_models/matched_weak.pth
    python load_model.py --model_path experiment_models/mismatched_weak_enc_strong_dec.pth
"""
import torch
import argparse
from modular_vae import ModularVAE


def load_and_inspect_model(model_path, device='cuda'):
    """Load a model and print its architecture"""
    print("=" * 80)
    print(f"Loading model from: {model_path}")
    print("=" * 80)
    
    model = torch.load(model_path, map_location=device)
    model = model.to(device)
    model.eval()
    
    print("\n" + "=" * 80)
    print("Model Architecture:")
    print("=" * 80)
    print(model)
    
    print("\n" + "=" * 80)
    print("Model Info:")
    print("=" * 80)
    print(f"  Decoder Type: {model.decoder_type}")
    if hasattr(model.decoder, 'decoder_type'):
        print(f"  Decoder Subtype: {model.decoder.decoder_type}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    
    print("\n" + "=" * 80)
    print("Model is ready to use!")
    print("=" * 80)
    print("\nExample usage:")
    print("  model.eval()")
    print("  with torch.no_grad():")
    print("      x_recon, kl, kl_obj = model(input)")
    print("      samples = model.sample(n_samples=64)")
    
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model')
    parser.add_argument('--device', type=str, default='cuda', help='Device to load model on')
    args = parser.parse_args()
    
    model = load_and_inspect_model(args.model_path, args.device)
