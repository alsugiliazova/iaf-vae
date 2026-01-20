#!/usr/bin/env python3
"""
Train matched VAE: Strong decoder + free bits (no IAF)
Approximately 3,685,028 parameters (matches IAF model)
"""

import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import time

from os.path import join
from tensorboardX import SummaryWriter
from torchvision.utils import save_image
from torchvision import datasets, transforms

from main import VAE
from utils import *

if __name__ == '__main__':
    # Matched VAE configuration (stronger decoder, no IAF)
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_blocks', type=int, default=5)      # Increased from 4
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--z_size', type=int, default=32)
    parser.add_argument('--h_size', type=int, default=64)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--free_bits', type=float, default=0.1)  # Free bits to prevent collapse
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--iaf', type=int, default=0)            # NO IAF
    args = parser.parse_args()

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    model = VAE(args).to(device)
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nTotal parameters: {total_params:,}')
    print(f'Target: ~3,685,028 (IAF model)\n')

    # Reproducibility
    set_seed(0)

    # Optimizer
    opt = torch.optim.Adamax(model.parameters(), lr=args.lr)

    # Data
    scale_inv = lambda x: x + 0.5
    ds_transforms = transforms.Compose([transforms.ToTensor(), lambda x: x - 0.5])
    kwargs = {'num_workers': 0, 'pin_memory': False, 'drop_last': True}

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=True, download=True, transform=ds_transforms),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, download=True, transform=ds_transforms),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    # Logging
    model_name = 'NB{}_D{}_Z{}_H{}_BS{}_FB{}_LR{}_IAF{}'.format(
        args.n_blocks, args.depth, args.z_size, args.h_size,
        args.batch_size, args.free_bits, args.lr, args.iaf)
    log_dir = join('runs', model_name)
    sample_dir = join(log_dir, 'samples')
    writer = SummaryWriter(log_dir=log_dir)
    maybe_create_dir(sample_dir)

    print_and_save_args(args, log_dir)
    print('logging into %s' % log_dir)
    best_test = float('inf')

    # Training
    print('starting training')
    total_start_time = time.time()
    epoch_times = []

    for epoch in range(args.n_epochs):
        epoch_start_time = time.time()
        model.train()
        train_log = reset_log()

        for batch_idx, (input, _) in enumerate(train_loader):
            input = input.to(device)
            x, kl, kl_obj = model(input)

            log_pxz = logistic_ll(x, model.dec_log_stdv, sample=input)
            loss = (kl_obj - log_pxz).sum() / x.size(0)
            elbo = (kl - log_pxz)
            bpd = elbo / (32 * 32 * 3 * np.log(2.))

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_log['kl'] += [kl.mean()]
            train_log['bpd'] += [bpd.mean()]
            train_log['elbo'] += [elbo.mean()]
            train_log['kl obj'] += [kl_obj.mean()]
            train_log['log p(x|z)'] += [log_pxz.mean()]

        for key, value in train_log.items():
            print_and_log_scalar(writer, 'train/%s' % key, value, epoch)
        
        # Log timing
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        elapsed_total = time.time() - total_start_time
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        writer.add_scalar('train/epoch_time', epoch_time, epoch)
        writer.add_scalar('train/total_time', elapsed_total, epoch)
        writer.add_scalar('train/avg_epoch_time', avg_epoch_time, epoch)
        
        print()

        model.eval()
        test_log = reset_log()

        with torch.no_grad():
            for batch_idx, (input, _) in enumerate(test_loader):
                input = input.to(device)
                x, kl, kl_obj = model(input)

                log_pxz = logistic_ll(x, model.dec_log_stdv, sample=input)
                loss = (kl_obj - log_pxz).sum() / x.size(0)
                elbo = (kl - log_pxz)
                bpd = elbo / (32 * 32 * 3 * np.log(2.))

                test_log['kl'] += [kl.mean()]
                test_log['bpd'] += [bpd.mean()]
                test_log['elbo'] += [elbo.mean()]
                test_log['kl obj'] += [kl_obj.mean()]
                test_log['log p(x|z)'] += [log_pxz.mean()]

            all_samples = model.cond_sample(input)
            out = torch.stack((x, input))
            out = out.transpose(1, 0).contiguous()
            out = out.view(-1, x.size(-3), x.size(-2), x.size(-1))

            all_samples += [x]
            all_samples = torch.stack(all_samples)
            all_samples = all_samples.transpose(1, 0)
            all_samples = all_samples.contiguous()
            all_samples = all_samples.view(-1, x.size(-3), x.size(-2), x.size(-1))

            save_image(scale_inv(all_samples), join(sample_dir, 'test_levels_{}.png'.format(epoch)), nrow=12)
            save_image(scale_inv(out), join(sample_dir, 'test_recon_{}.png'.format(epoch)), nrow=12)
            save_image(scale_inv(model.sample(64)), join(sample_dir, 'sample_{}.png'.format(epoch)), nrow=8)

        for key, value in test_log.items():
            print_and_log_scalar(writer, 'test/%s' % key, value, epoch)
        print()

        current_test = sum(test_log['bpd']) / len(test_log['bpd'])
        if current_test < best_test:
            best_test = current_test
            print('saving best model')
            torch.save(model.state_dict(), join(log_dir, 'best_model.pth'))

        # Calculate remaining time
        remaining_epochs = args.n_epochs - (epoch + 1)
        estimated_remaining = avg_epoch_time * remaining_epochs
        writer.add_scalar('train/estimated_remaining_time', estimated_remaining, epoch)

        print(f'Epoch {epoch+1}/{args.n_epochs} - Time: {epoch_time:.2f}s | '
              f'Avg: {avg_epoch_time:.2f}s | '
              f'Elapsed: {elapsed_total/60:.1f}m | '
              f'Est. remaining: {estimated_remaining/60:.1f}m')
        print()

    total_time = time.time() - total_start_time
    print('='*60)
    print('Training completed!')
    print(f'Total time: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)')
    print(f'Average per epoch: {sum(epoch_times)/len(epoch_times):.2f} seconds ({sum(epoch_times)/len(epoch_times)/60:.2f} minutes)')
    print('='*60)
