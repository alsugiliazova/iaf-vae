"""
Training script for ModularVAE with different decoder types.

Usage:
    # Train weak decoder model
    python train_modular.py --decoder_type weak --decoder_subtype mlp --n_epochs 50
    
    # Train strong decoder model
    python train_modular.py --decoder_type strong --n_epochs 50
"""
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from os.path import join
from tensorboardX import SummaryWriter
from torchvision.utils import save_image
from collections import OrderedDict as OD 
from torchvision import datasets, transforms, utils

from modular_vae import ModularVAE
from utils import *


if __name__ == '__main__':
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--decoder_type', type=str, default='weak', choices=['weak', 'strong'],
                        help='Type of decoder: weak (MLP/small conv) or strong (ResNet)')
    parser.add_argument('--decoder_subtype', type=str, default='mlp', choices=['mlp', 'small_conv'],
                        help='Subtype for weak decoder: mlp or small_conv')
    parser.add_argument('--n_blocks', type=int, default=20)
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--z_size', type=int, default=32)
    parser.add_argument('--h_size', type=int, default=160)
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--free_bits', type=float, default=0.1)
    parser.add_argument('--iaf', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'mnist'])
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory for dataset')
    args = parser.parse_args()

    # create model and ship to GPU
    model = ModularVAE(args, decoder_type=args.decoder_type, decoder_subtype=args.decoder_subtype).cuda()
    
    print("=" * 80)
    print("Model Architecture:")
    print("=" * 80)
    print(model)
    print("\n" + "=" * 80)
    print(f"Decoder Type: {args.decoder_type}")
    if args.decoder_type == 'weak':
        print(f"Decoder Subtype: {args.decoder_subtype}")
    print("=" * 80 + "\n")

    # reproducibility
    set_seed(0)

    opt = torch.optim.Adamax(model.parameters(), lr=args.lr)

    # create datasets / dataloaders
    scale_inv = lambda x : x + 0.5
    ds_transforms = transforms.Compose([transforms.ToTensor(), lambda x : x - 0.5])
    kwargs = {'num_workers':1, 'pin_memory':True, 'drop_last':True}

    if args.dataset == 'cifar10':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.data_dir, train=True, download=True, transform=ds_transforms),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.data_dir, train=False, download=True, transform=ds_transforms),
            batch_size=args.batch_size, shuffle=True, **kwargs)
    elif args.dataset == 'mnist':
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(args.data_dir, train=True, download=True, transform=ds_transforms),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(args.data_dir, train=False, download=True, transform=ds_transforms),
            batch_size=args.batch_size, shuffle=True, **kwargs)

    # spawn writer
    decoder_name = f"{args.decoder_type}_{args.decoder_subtype}" if args.decoder_type == 'weak' else args.decoder_type
    model_name = 'NB{}_D{}_Z{}_H{}_BS{}_FB{}_LR{}_IAF{}_DEC{}'.format(
        args.n_blocks, args.depth, args.z_size, args.h_size, 
        args.batch_size, args.free_bits, args.lr, args.iaf, decoder_name)
    
    model_name = 'test' if args.debug else model_name
    log_dir = join('runs', model_name)
    sample_dir = join(log_dir, 'samples')
    writer = SummaryWriter(log_dir=log_dir)
    maybe_create_dir(sample_dir)

    print_and_save_args(args, log_dir)
    print('logging into %s' % log_dir)
    best_test = float('inf')

    print('starting training')
    for epoch in range(args.n_epochs):
        model.train()
        train_log = reset_log()

        for batch_idx, (input,_) in enumerate(train_loader):
            input = input.cuda()
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
        print()
        
        model.eval()
        test_log = reset_log()

        with torch.no_grad():
            for batch_idx, (input,_) in enumerate(test_loader):
                input = input.cuda()
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
                
            # Save samples
            save_image(scale_inv(model.sample(64)), join(sample_dir, 'sample_{}.png'.format(epoch)), nrow=8)
            
            # Save reconstructions
            out = torch.stack((x[:8], input[:8]))
            out = out.transpose(1,0).contiguous()
            out = out.view(-1, x.size(-3), x.size(-2), x.size(-1))
            save_image(scale_inv(out), join(sample_dir, 'test_recon_{}.png'.format(epoch)), nrow=8)

        for key, value in test_log.items():
            print_and_log_scalar(writer, 'test/%s' % key, value, epoch)
        print()
        
        current_test = sum(test_log['bpd']) / len(test_log['bpd'])
        if current_test < best_test:
            best_test = current_test
            print('saving best model')
            torch.save(model.state_dict(), join(log_dir, 'best_model.pth'))
            # Also save full model for easier loading
            torch.save(model, join(log_dir, 'best_model_full.pth'))
    
    print(f"\nTraining complete! Best test BPD: {best_test:.4f}")
    print(f"Model saved to: {join(log_dir, 'best_model_full.pth')}")
