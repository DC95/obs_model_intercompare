

#CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node=1 eval_features_msgrandom108.py --pretrained_weight /p/scratch/exaww/chatterjee1/nn_obs/continuous/checkpoint.pth

#arch: vit_small
#batch_size_per_gpu: 512
#checkpoint_key: teacher
#data_paths: ['/p/project1/exaww/chatterjee1/dataset/warmworld_datasets/msgobs_108_randcrops_2020.nc', '/p/project1/exaww/chatterjee1/dataset/warmworld_datasets/msgobs_108_randcrops_2021.nc', '/p/project1/exaww/chatterjee1/dataset/warmworld_datasets/msgobs_108_randcrops_2022.nc', '/p/project1/exaww/chatterjee1/dataset/warmworld_datasets/msgobs_108_randcrops_2023.nc', '/p/project1/exaww/chatterjee1/dataset/warmworld_datasets/msgobs_108_randcrops_2024.nc']
#dist_url: env://
#dump_features: /p/scratch/exaww/chatterjee1/nn_obs/all_nc/features_obs/
#gpu: 0
#load_features: None
#local_rank: 0
#mean_108: [270.559]
#nb_knn: [10, 20, 100, 200]
#num_workers: 16
#patch_size: 8
#pretrained_weights: /p/scratch/exaww/chatterjee1/nn_obs/all_nc/checkpoint.pth
#rank: 0
#std_108: [17.608]
#temperature: 0.07
#use_cuda: True
#variables: ['sample_data']
#world_size: 1
#Loaded 383928 samples from 5 NetCDF files.
#Model vit_small 8x8 built.
#Take key teacher in provided checkpoint dict
#Pretrained weights found at /p/scratch/exaww/chatterjee1/nn_obs/all_nc/checkpoint.pth and loaded with msg: _IncompatibleKeys(missing_keys=[], unexpected_keys=['head.mlp.0.weight', 'head.mlp.0.bias', 'head.mlp.2.weight', 'head.mlp.2.bias', 'head.mlp.4.weight', 'head.mlp.4.bias', 'head.last_layer.weight_g', 'head.last_layer.weight_v'])
#Extracting features for train set...
#Storing features into tensor of shape torch.Size([383928, 384])

import os
import sys
import argparse

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models

import utils_msg108
import vision_transformer as vits
import h5py
from netCDF4 import Dataset as NC
from contextlib import nullcontext
import warnings
warnings.filterwarnings("ignore")

def get_args_parser_msg():

    parser = argparse.ArgumentParser('Extracting features with trained MSG-ATMOS', add_help=False)
    parser.add_argument('--batch_size_per_gpu', default=512, type=int, help='Per-GPU batch-size')
    parser.add_argument('--nb_knn', default=[10, 20, 100, 200], nargs='+', type=int,
        help='Number of NN to use. 20 is usually working the best.')
    parser.add_argument('--temperature', default=0.07, type=float,
        help='Temperature used in the voting coefficient')
    parser.add_argument('--mean_108', default=[270.559], type=float,
        help='mean of 10.8 um temperature used during normalization') # 266.87 for icon # 270.526 for obs
    parser.add_argument('--std_108', default=[17.608], type=float,
        help='standard deviation of 10.8 um T used during normalization') # 12.64 for icon # 17.901 for obs
    parser.add_argument('--variables', type=str, nargs='+', default=['sample_data'],
        help="""Variables to read from NetCDF, e.g. --variables sample_data""") # for obs: sample_data
    parser.add_argument('--pretrained_weights', default='/p/scratch/exaww/chatterjee1/nn_obs/all_nc/checkpoint.pth', type=str, help="Path to pretrained weights to evaluate.") #dino_256_trial
    parser.add_argument('--use_cuda', default=True, type=utils_msg108.bool_flag,
        help="Should we store the features on GPU? We recommend setting this to False if you encounter OOM")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--dump_features', default='/p/scratch/exaww/chatterjee1/nn_obs/all_nc/features_obs/', help='Path where to save computed features, empty for no saving')
    parser.add_argument('--load_features', default=None, help="""If the features have
        already been computed, where to find them.""")
    parser.add_argument('--num_workers', default=16, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument(
        '--data_paths',
        nargs='+',
        default=[
            '/p/project1/exaww/chatterjee1/dataset/warmworld_datasets/msgobs_108_randcrops_2020.nc',
            '/p/project1/exaww/chatterjee1/dataset/warmworld_datasets/msgobs_108_randcrops_2021.nc',
            '/p/project1/exaww/chatterjee1/dataset/warmworld_datasets/msgobs_108_randcrops_2022.nc',
            '/p/project1/exaww/chatterjee1/dataset/warmworld_datasets/msgobs_108_randcrops_2023.nc',
            '/p/project1/exaww/chatterjee1/dataset/warmworld_datasets/msgobs_108_randcrops_2024.nc',
        ],
        help='One or more NetCDF files (accepts globs).'
    ) 
    return parser

class FastNCDataset(torch.utils.data.Dataset):
    def __init__(self, nc_paths, variable_names, transform=None, keep_open=True):
        self.paths = list(nc_paths)
        self.vars = list(variable_names)
        self.transform = transform
        self.keep_open = keep_open

        self.handles = [NC(p, "r") for p in self.paths] if keep_open else None
        self.index_map, self.lengths = [], []
        for fi, p in enumerate(self.paths):
            ds = self.handles[fi] if self.handles else NC(p, "r")
            n = ds.dimensions["sample"].size
            self.lengths.append(n)
            self.index_map.extend((fi, i) for i in range(n))
            if not self.keep_open: ds.close()

        self.total = sum(self.lengths)

        # disable mask/scale once (faster)
        if self.keep_open:
            for ds in self.handles:
                for v in self.vars:
                    ds.variables[v].set_auto_maskandscale(False)

    def __len__(self): return self.total

    def __getitem__(self, idx):
        fi, li = self.index_map[idx]
        ds = self.handles[fi] if self.keep_open else NC(self.paths[fi], "r")
        sample = {}
        for v in self.vars:
            var = ds.variables[v]
            if not self.keep_open:
                var.set_auto_maskandscale(False)
            arr = var[li, ...]                 # (H, W)
            sample[v] = torch.as_tensor(arr, dtype=torch.float32).unsqueeze(0)  # (1,H,W)
        if not self.keep_open:
            ds.close()
        if self.transform:
            sample = self.transform(sample)
        return sample, idx

class MSGAugmentation_featureextraction(object):
    def __init__(self, mean_108, std_108):
        self.normalize = pth_transforms.Normalize(mean=mean_108, std=std_108)

    def __call__(self, sample):
        # Assuming 'image' is a key in the sample dictionary
        image = sample['sample_data'] # for icon: model_108, #for obs: sample_data  

        if image.ndim == 2:
            image = image.unsqueeze(0)  # Add a channel dimension

        # Normalize
        crop = self.normalize(image)
    
        return crop 


@torch.no_grad()
def extract_feature_pipeline(args):
    # ============ preparing data ... ============
    transform = MSGAugmentation_featureextraction(
        mean_108=args.mean_108,
        std_108=args.std_108,
    )

    dataset_train = FastNCDataset(args.data_paths,args.variables,transform=transform, keep_open=True)
    sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=False,
    )

    print(f"Loaded {len(dataset_train)} samples from {len(dataset_train.paths)} NetCDF files.") 

    # ============ building network ... ============
    if "vit" in args.arch:
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    else:
        print(f"Architecture {args.arch} not supported")
        sys.exit(1)
    model.cuda()
    utils_msg108.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    model.eval()

    # ============ extract features ... ============
    print("Extracting features for train set...")
    train_features = extract_features(model, data_loader_train, args.use_cuda)

    if utils_msg108.get_rank() == 0:
        train_features = nn.functional.normalize(train_features, dim=1, p=2)

    # Use the dataset indices as labels
    train_labels = torch.tensor(range(len(dataset_train)), dtype=torch.long)

    # save features and labels
    if args.dump_features and dist.get_rank() == 0:
        torch.save(train_features.cpu(), os.path.join(args.dump_features, "trainfeat_multiyear.pth"))
        torch.save(train_labels.cpu(), os.path.join(args.dump_features, "trainlabels_multiyear.pth"))
    return train_features, train_labels



@torch.no_grad()
def extract_features(model, data_loader, use_cuda=True, multiscale=False):
    metric_logger = utils_msg108.MetricLogger(delimiter="  ")
    features = None
    for samples, index in metric_logger.log_every(data_loader, 10):
        samples = samples.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        if multiscale:
            feats = utils_msg108.multi_scale(samples, model)
        else:
            feats = model(samples).clone()

        # init storage feature matrix
        if dist.get_rank() == 0 and features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            if use_cuda:
                features = features.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")

        # get indexes from all processes
        y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)

        # share features between processes
        feats_all = torch.empty(
            dist.get_world_size(),
            feats.size(0),
            feats.size(1),
            dtype=feats.dtype,
            device=feats.device,
        )
        output_l = list(feats_all.unbind(0))
        output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()

        # update storage feature matrix
        if dist.get_rank() == 0:
            if use_cuda:
                features.index_copy_(0, index_all, torch.cat(output_l))
            else:
                features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
    return features

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('Extracting features with trained MSG-ATMOS', parents=[get_args_parser_msg()])
    args = parser.parse_args()
    utils_msg108.init_distributed_mode(args)
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True
    train_features,  train_labels = extract_feature_pipeline(args) #, test_features,test_labels

