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

from layers import IAFLayer
from utils import *

# Model definition
# ----------------------------------------------------------------------------------------------
class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.register_parameter('h', torch.nn.Parameter(torch.zeros(args.h_size)))
        self.register_parameter('dec_log_stdv', torch.nn.Parameter(torch.Tensor([0.])))
        
        # Dataset-specific settings
        self.dataset = args.dataset
        if args.dataset == 'mnist':
            self.in_channels = 1
            self.out_channels = 1
            self.image_size = 32 if args.pad_mnist else 28  # Pad to 32x32 or keep 28x28
            self.use_bernoulli = True
        else:  # cifar10
            self.in_channels = 3
            self.out_channels = 3
            self.image_size = 32
            self.use_bernoulli = False

        layers = []
        # build network
        for i in range(args.depth):
            layer = []

            for j in range(args.n_blocks):
                downsample = (i > 0) and (j == 0)
                layer += [IAFLayer(args, downsample)]

            layers += [nn.ModuleList(layer)]

        self.layers = nn.ModuleList(layers) 
        
        # First conv: input -> hidden
        self.first_conv = nn.Conv2d(self.in_channels, args.h_size, 4, 2, 1)
        # Last conv: hidden -> output
        self.last_conv = nn.ConvTranspose2d(args.h_size, self.out_channels, 4, 2, 1)

    def forward(self, input):
        # assumes input is \in [-0.5, 0.5] 
        x = self.first_conv(input)
        kl, kl_obj = 0., 0.

        h = self.h.view(1, -1, 1, 1)

        for layer in self.layers:
            for sub_layer in layer:
                x = sub_layer.up(x)

        h = h.expand_as(x)
        self.hid_shape = x[0].size()

        for layer in reversed(self.layers):
            for sub_layer in reversed(layer):
                h, curr_kl, curr_kl_obj = sub_layer.down(h)
                kl     += curr_kl
                kl_obj += curr_kl_obj

        x = F.elu(h)
        x = self.last_conv(x)
        
        if self.use_bernoulli:
            # For MNIST: output is logits, will apply sigmoid later
            # No clamping needed for Bernoulli
            pass
        else:
            # For CIFAR-10: clamp to [-0.5 + 1/512, 0.5 - 1/512]
            x = x.clamp(min=-0.5 + 1. / 512., max=0.5 - 1. / 512.)

        return x, kl, kl_obj


    def sample(self, n_samples=64):
        h = self.h.view(1, -1, 1, 1)
        h = h.expand((n_samples, *self.hid_shape))
        
        for layer in reversed(self.layers):
            for sub_layer in reversed(layer):
                h, _, _ = sub_layer.down(h, sample=True)

        x = F.elu(h)
        x = self.last_conv(x)
        
        if self.use_bernoulli:
            return x  # No clamping for Bernoulli
        else:
            return x.clamp(min=-0.5 + 1. / 512., max=0.5 - 1. / 512.)
    
    
    def cond_sample(self, input):
        # assumes input is \in [-0.5, 0.5] 
        x = self.first_conv(input)
        kl, kl_obj = 0., 0.

        h = self.h.view(1, -1, 1, 1)

        for layer in self.layers:
            for sub_layer in layer:
                x = sub_layer.up(x)

        h = h.expand_as(x)
        self.hid_shape = x[0].size()

        outs = []

        current = 0
        for i, layer in enumerate(reversed(self.layers)):
            for j, sub_layer in enumerate(reversed(layer)):
                h, curr_kl, curr_kl_obj = sub_layer.down(h)
                
                h_copy = h
                again = 0
                # now, sample the rest of the way:
                for layer_ in reversed(self.layers):
                    for sub_layer_ in reversed(layer_):
                        if again > current:
                            h_copy, _, _ = sub_layer_.down(h_copy, sample=True)
                        
                        again += 1
                        
                x = F.elu(h_copy)
                x = self.last_conv(x)
                if self.use_bernoulli:
                    pass  # No clamping for Bernoulli
                else:
                    x = x.clamp(min=-0.5 + 1. / 512., max=0.5 - 1. / 512.)
                outs += [x]

                current += 1

        return outs
        
# Main
# ----------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--n_blocks', type=int, default=4)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--z_size', type=int, default=32)
    parser.add_argument('--h_size', type=int, default=64)
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--free_bits', type=float, default=0.1)
    parser.add_argument('--iaf', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'mnist'],
                        help='Dataset to use: cifar10 or mnist')
    parser.add_argument('--pad_mnist', action='store_true',
                        help='Pad MNIST to 32x32 (matches OpenAI implementation)')
    args = parser.parse_args()

    # create model and ship to device (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    model = VAE(args).to(device)
    print(model)

    # reproducibility is da best
    set_seed(0)

    opt = torch.optim.Adamax(model.parameters(), lr=args.lr)

    # Helper functions for transforms (must be top-level for pickling)
    def binarize(x):
        """Binarize MNIST: threshold at 0.5"""
        return (x > 0.5).float()
    
    def shift_cifar(x):
        """Shift CIFAR-10 to [-0.5, 0.5]"""
        return x - 0.5
    
    def scale_inv_mnist(x):
        """Scale inverse for MNIST (Bernoulli output)"""
        return torch.clamp(x, 0, 1)
    
    def scale_inv_cifar(x):
        """Scale inverse for CIFAR-10"""
        return x + 0.5
    
    # create datasets / dataloaders
    data_dir = './data'  # Standard location for datasets
    # Use num_workers=0 on macOS to avoid multiprocessing issues, or use num_workers=1 with proper functions
    kwargs = {'num_workers':0, 'pin_memory':False, 'drop_last':True}  # Changed to avoid pickling issues
    
    if args.dataset == 'mnist':
        # MNIST: 28x28 grayscale, binarized
        # Pad to 32x32 if requested (matches OpenAI implementation)
        if args.pad_mnist:
            pad_transform = transforms.Compose([
                transforms.Pad(2),  # Pad 28x28 -> 32x32
                transforms.ToTensor(),
                binarize  # Binarize: threshold at 0.5
            ])
        else:
            pad_transform = transforms.Compose([
                transforms.ToTensor(),
                binarize  # Binarize: threshold at 0.5
            ])
        
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_dir, train=True, download=True, transform=pad_transform),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_dir, train=False, download=True, transform=pad_transform),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        
        scale_inv = scale_inv_mnist
        image_size = 32 if args.pad_mnist else 28
        
    else:  # cifar10
        scale_inv = scale_inv_cifar
        ds_transforms = transforms.Compose([transforms.ToTensor(), shift_cifar])
        
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(data_dir, train=True, download=True, transform=ds_transforms),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(data_dir, train=False, download=True, transform=ds_transforms),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        
        image_size = 32

    # spawn writer
    model_name = 'NB{}_D{}_Z{}_H{}_BS{}_FB{}_LR{}_IAF{}'.format(args.n_blocks, args.depth, args.z_size, args.h_size, 
                                                                args.batch_size, args.free_bits, args.lr, args.iaf)

    model_name = 'test' if args.debug else model_name
    log_dir    = join('runs', model_name)
    sample_dir = join(log_dir, 'samples')
    writer     = SummaryWriter(log_dir=log_dir)
    maybe_create_dir(sample_dir)

    print_and_save_args(args, log_dir)
    print('logging into %s' % log_dir)
    maybe_create_dir(sample_dir)
    best_test = float('inf')


    print('starting training')
    print(f'Training batches per epoch: {len(train_loader)}')
    print(f'Test batches per epoch: {len(test_loader)}')
    
    if len(train_loader) == 0:
        raise ValueError("Train loader is empty! Check batch_size and dataset.")
    if len(test_loader) == 0:
        raise ValueError("Test loader is empty! Check batch_size and dataset.")
    
    for epoch in range(args.n_epochs):
        model.train()
        train_log = reset_log()

        for batch_idx, (input,_) in enumerate(train_loader):
            try:
                input = input.to(device)
                x, kl, kl_obj = model(input)
            except Exception as e:
                print(f'Error in forward pass at epoch {epoch}, batch {batch_idx}: {e}')
                import traceback
                traceback.print_exc()
                raise

            if model.use_bernoulli:
                # MNIST: use Bernoulli likelihood
                from utils import bernoulli_ll
                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(x)
                log_pxz = bernoulli_ll(probs, sample=input)
                loss = (kl_obj - log_pxz).sum() / x.size(0)
                elbo = (kl - log_pxz)
                # Bits per dimension: negative ELBO / (pixels * log(2))
                n_pixels = model.image_size * model.image_size * 1  # 1 channel for MNIST
                bpd = elbo / (n_pixels * np.log(2.))
            else:
                # CIFAR-10: use logistic likelihood
                log_pxz = logistic_ll(x, model.dec_log_stdv, sample=input)
                loss = (kl_obj - log_pxz).sum() / x.size(0)
                elbo = (kl - log_pxz)
                bpd = elbo / (32 * 32 * 3 * np.log(2.))
         
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_log['kl']         += [kl.mean()]
            train_log['bpd']        += [bpd.mean()]
            train_log['elbo']       += [elbo.mean()]
            train_log['kl obj']     += [kl_obj.mean()]
            train_log['log p(x|z)'] += [log_pxz.mean()]

        if len(train_log) > 0:
            for key, value in train_log.items():
                print_and_log_scalar(writer, 'train/%s' % key, value, epoch)
        else:
            print(f'Epoch {epoch}: No training batches processed!')
        print()
        
        model.eval()
        test_log = reset_log()

        with torch.no_grad():
            for batch_idx, (input,_) in enumerate(test_loader):
                try:
                    input = input.to(device)
                    x, kl, kl_obj = model(input)
                except Exception as e:
                    print(f'Error in test forward pass at epoch {epoch}, batch {batch_idx}: {e}')
                    import traceback
                    traceback.print_exc()
                    raise
            
                if model.use_bernoulli:
                    # MNIST: use Bernoulli likelihood
                    from utils import bernoulli_ll
                    probs = torch.sigmoid(x)
                    log_pxz = bernoulli_ll(probs, sample=input)
                    loss = (kl_obj - log_pxz).sum() / x.size(0)
                    elbo = (kl - log_pxz)
                    n_pixels = model.image_size * model.image_size * 1
                    bpd = elbo / (n_pixels * np.log(2.))
                else:
                    # CIFAR-10: use logistic likelihood
                    log_pxz = logistic_ll(x, model.dec_log_stdv, sample=input)
                    loss = (kl_obj - log_pxz).sum() / x.size(0)
                    elbo = (kl - log_pxz)
                    bpd = elbo / (32 * 32 * 3 * np.log(2.))
                
                test_log['kl']         += [kl.mean()]
                test_log['bpd']        += [bpd.mean()]
                test_log['elbo']       += [elbo.mean()]
                test_log['kl obj']     += [kl_obj.mean()]
                test_log['log p(x|z)'] += [log_pxz.mean()]
                
            all_samples = model.cond_sample(input)
            # save reconstructions
            out = torch.stack((x, input))               # 2, bs, 3, 32, 32
            out = out.transpose(1,0).contiguous()       # bs, 2, 3, 32, 32
            out = out.view(-1, x.size(-3), x.size(-2), x.size(-1))
           
            all_samples += [x]
            all_samples = torch.stack(all_samples)     # L, bs, 3, 32, 32
            all_samples = all_samples.transpose(1,0)
            all_samples = all_samples.contiguous()     # bs, L, 3, 32, 32
            all_samples = all_samples.view(-1, x.size(-3), x.size(-2), x.size(-1))

            save_image(scale_inv(all_samples), join(sample_dir, 'test_levels_{}.png'.format(epoch)), nrow=12)
            save_image(scale_inv(out), join(sample_dir, 'test_recon_{}.png'.format(epoch)), nrow=12)
            save_image(scale_inv(model.sample(64)), join(sample_dir, 'sample_{}.png'.format(epoch)), nrow=8)
            

        if len(test_log) > 0:
            for key, value in test_log.items():
                print_and_log_scalar(writer, 'test/%s' % key, value, epoch)
        else:
            print(f'Epoch {epoch}: No test batches processed!')
        print()
        
        current_test = sum(test_log['bpd']) / len(test_log['bpd']) if len(test_log['bpd']) > 0 else float('inf')
        if current_test < best_test:
            best_test = current_test
            print('saving best model')
            torch.save(model.state_dict(), join(log_dir, 'best_model.pth'))
