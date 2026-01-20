import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import time

from os.path import join
from tensorboardX import SummaryWriter
from torchvision.utils import save_image
from collections import OrderedDict as OD 
from torchvision import datasets, transforms, utils

from layers import IAFLayer, AutoregressivePrior
from utils  import * 

# Model definition
# ----------------------------------------------------------------------------------------------
class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.register_parameter('h', torch.nn.Parameter(torch.zeros(args.h_size)))
        self.register_parameter('dec_log_stdv', torch.nn.Parameter(torch.Tensor([0.])))
        
        self.args = args
        self.ar_prior = getattr(args, 'ar_prior', 0)
        
        # When using autoregressive prior, disable IAF (use diagonal posterior)
        if self.ar_prior:
            args.iaf = 0

        layers = []
        # build network
        for i in range(args.depth):
            layer = []

            for j in range(args.n_blocks):
                downsample = (i > 0) and (j == 0)
                layer += [IAFLayer(args, downsample)]

            layers += [nn.ModuleList(layer)]

        self.layers = nn.ModuleList(layers) 
        
        self.first_conv = nn.Conv2d(3, args.h_size, 4, 2, 1)
        self.last_conv = nn.ConvTranspose2d(args.h_size, 3, 4, 2, 1)
        
        # Initialize autoregressive prior if enabled
        if self.ar_prior:
            # We need to determine the total z dimension after all layers
            # This will be set during first forward pass
            self.ar_prior_module = None
            self.total_z_dim = None

    def forward(self, input):
        # assumes input is \in [-0.5, 0.5] 
        x = self.first_conv(input)
        
        # Initialize kl and kl_obj - will be set based on ar_prior mode
        if self.ar_prior:
            kl = None
            kl_obj = None
        else:
            kl, kl_obj = 0., 0.

        h = self.h.view(1, -1, 1, 1)

        for layer in self.layers:
            for sub_layer in layer:
                x = sub_layer.up(x)

        h = h.expand_as(x)
        self.hid_shape = x[0].size()

        # Collect all z values if using autoregressive prior
        all_z = [] if self.ar_prior else None
        all_logqs = [] if self.ar_prior else None
        
        for layer in reversed(self.layers):
            for sub_layer in reversed(layer):
                if self.ar_prior:
                    # Store z and logq before computing KL
                    # We'll compute KL with autoregressive prior after collecting all z
                    h, curr_kl, curr_kl_obj, z, logqs = sub_layer.down(h, return_z=True)
                    all_z.append(z)
                    all_logqs.append(logqs)
                    # Don't add curr_kl to kl yet - we'll recompute with AR prior
                else:
                    h, curr_kl, curr_kl_obj = sub_layer.down(h)
                    kl     += curr_kl
                    kl_obj += curr_kl_obj

        # If using autoregressive prior, compute global log p(z) and recompute KL
        if self.ar_prior:
            # Store original shapes for free bits application
            z_shapes = []
            logqs_shapes = []
            
            # Concatenate all z values: each z has shape (B, z_size, H, W)
            # Flatten spatial dimensions and concatenate across layers
            z_flat_list = []
            logqs_flat_list = []
            for z, logqs in zip(all_z, all_logqs):
                B, C, H, W = z.size()
                z_shapes.append((B, C, H, W))
                logqs_shapes.append((B, C, H, W))
                z_flat = z.view(B, -1)  # (B, z_size * H * W)
                logqs_flat = logqs.view(B, -1)  # (B, z_size * H * W)
                z_flat_list.append(z_flat)
                logqs_flat_list.append(logqs_flat)
            
            z_all_flat = torch.cat(z_flat_list, dim=1)  # (B, total_z_dim)
            logqs_all_flat = torch.cat(logqs_flat_list, dim=1)  # (B, total_z_dim)
            
            # Initialize autoregressive prior if needed
            if self.ar_prior_module is None:
                total_z_dim = z_all_flat.size(1)
                self.total_z_dim = total_z_dim
                # Use similar hidden size to IAF (args.h_size)
                hidden_dim = self.args.h_size
                self.ar_prior_module = AutoregressivePrior(
                    z_dim=total_z_dim,
                    hidden_dim=hidden_dim,
                    n_layers=2
                ).to(z_all_flat.device)
            
            # Compute log p(z) using autoregressive prior
            logps_ar, means_ar, log_stds_ar = self.ar_prior_module(z_all_flat)
            
            # Compute log p(z) per dimension
            stds_ar = torch.exp(log_stds_ar)
            dist_ar = D.Normal(means_ar, stds_ar)
            logps_per_dim = dist_ar.log_prob(z_all_flat)  # (B, total_z_dim)
            
            # Compute KL per dimension: log q(z|x) - log p(z)
            kl_per_dim_flat = logqs_all_flat - logps_per_dim  # (B, total_z_dim)
            
            # Reshape KL back to original structure to apply free bits correctly
            # Match baseline: free bits applied per z_size dimension (across spatial locations)
            kl_per_layer = []
            start_idx = 0
            for (B, C, H, W) in z_shapes:
                layer_size = C * H * W
                kl_layer_flat = kl_per_dim_flat[:, start_idx:start_idx + layer_size]
                kl_layer = kl_layer_flat.view(B, C, H, W)  # (B, z_size, H, W)
                kl_per_layer.append(kl_layer)
                start_idx += layer_size
            
            # Apply free bits exactly as in baseline: per z_size dimension
            kl_obj = 0.
            kl_total = 0.
            for kl_layer in kl_per_layer:
                # kl_layer shape: (B, z_size, H, W)
                # Sum over spatial dims: (B, z_size, H, W) -> (B, z_size)
                kl_per_z = kl_layer.sum(dim=(-2, -1))  # (B, z_size)
                
                # Average over batch and clamp: match baseline behavior
                kl_obj_per_z = kl_per_z.mean(dim=0, keepdim=True)  # (1, z_size)
                kl_obj_per_z = kl_obj_per_z.clamp(min=self.args.free_bits)  # (1, z_size)
                kl_obj_per_z = kl_obj_per_z.expand(kl_per_z.size(0), -1)  # (B, z_size)
                kl_obj += kl_obj_per_z.sum(dim=1)  # (B,)
                
                # Total KL for logging: sum over all dims
                kl_total += kl_per_z.sum(dim=1)  # (B,)
            
            kl = kl_total

        x = F.elu(h)
        x = self.last_conv(x)
        
        x = x.clamp(min=-0.5 + 1. / 512., max=0.5 - 1. / 512.)

        return x, kl, kl_obj


    def sample(self, n_samples=64):
        # For now, use standard sampling even with autoregressive prior
        # TODO: Implement proper autoregressive prior sampling
        # (This requires sampling z sequentially and reshaping to spatial structure)
        h = self.h.view(1, -1, 1, 1)
        h = h.expand((n_samples, *self.hid_shape))
        
        for layer in reversed(self.layers):
            for sub_layer in reversed(layer):
                h, _, _ = sub_layer.down(h, sample=True)

        x = F.elu(h)
        x = self.last_conv(x)
        
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
    parser.add_argument('--ar_prior', type=int, default=0, help='Use autoregressive prior (Model C). When enabled, IAF is disabled.')
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    # create model and ship to GPU
    model = VAE(args).cuda()
    print(model)

    # reproducibility is da best
    set_seed(0)

    opt = torch.optim.Adamax(model.parameters(), lr=args.lr)

    # create datasets / dataloaders
    scale_inv = lambda x : x + 0.5
    ds_transforms = transforms.Compose([transforms.ToTensor(), lambda x : x - 0.5])
    kwargs = {'num_workers':1, 'pin_memory':True, 'drop_last':True}

    train_loader = torch.utils.data.DataLoader(datasets.CIFAR10('../cl-pytorch/data', train=True, 
        download=True, transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader  = torch.utils.data.DataLoader(datasets.CIFAR10('../cl-pytorch/data', train=False, 
        download=True, transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)

    # spawn writer
    model_name = 'NB{}_D{}_Z{}_H{}_BS{}_FB{}_LR{}_IAF{}_AR{}'.format(args.n_blocks, args.depth, args.z_size, args.h_size, 
                                                                args.batch_size, args.free_bits, args.lr, args.iaf, args.ar_prior)

    model_name = 'test' if args.debug else model_name
    log_dir    = join('runs', model_name)
    sample_dir = join(log_dir, 'samples')
    writer     = SummaryWriter(log_dir=log_dir)
    maybe_create_dir(sample_dir)

    print_and_save_args(args, log_dir)
    print('logging into %s' % log_dir)
    best_test = float('inf')


    print('starting training')
    total_start_time = time.time()
    epoch_times = []
    
    for epoch in range(args.n_epochs):
        epoch_start_time = time.time()
        model.train()
        train_log = reset_log()

        for batch_idx, (input,_) in enumerate(train_loader):

            input = input.cuda()
            x, kl, kl_obj = model(input)

            log_pxz = logistic_ll(x, model.dec_log_stdv, sample=input)
            loss = (kl_obj - log_pxz).sum() / x.size(0)
            elbo = (kl     - log_pxz)
            bpd  = elbo / (32 * 32 * 3 * np.log(2.))
         
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_log['kl']         += [kl.mean()]
            train_log['bpd']        += [bpd.mean()]
            train_log['elbo']       += [elbo.mean()]
            train_log['kl obj']     += [kl_obj.mean()]
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
                elbo = (kl     - log_pxz)
                bpd  = elbo / (32 * 32 * 3 * np.log(2.))
                
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
            

        for key, value in test_log.items():
            print_and_log_scalar(writer, 'test/%s' % key, value, epoch)
        print()
        
        current_test = sum(test_log['bpd']) / len(test_log['bpd'])
        if current_test < best_test:
            best_test = current_test
            print('saving best model')
            torch.save(model.state_dict(), join(log_dir, 'best_model.pth'))
        
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        elapsed_total = time.time() - total_start_time
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = args.n_epochs - (epoch + 1)
        estimated_remaining = avg_epoch_time * remaining_epochs
        
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