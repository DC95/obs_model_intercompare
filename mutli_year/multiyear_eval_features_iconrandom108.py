

#CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node=1 eval_features_iconrandom108.py --pretrained_weight /p/scratch/exaww/chatterjee1/nn_obs/continuous/checkpoint.pth

#arch: vit_small
#batch_size_per_gpu: 512
#checkpoint_key: teacher
#data_path: /p/project1/exaww/chatterjee1/dataset/msgobs_108_randcrops_icon.h5
#dist_url: env://
#dump_features: /p/project/exaww/chatterjee1/mcspss_continuous/analysis/features_icon/
#gpu: 0
#load_features: None
#local_rank: 0
#mean_ze: [266.87]
#nb_knn: [10, 20, 100, 200]
#num_workers: 16
#patch_size: 16
#pretrained_weights: /p/scratch/exaww/chatterjee1/nn_obs/continuous/checkpoint.pth
#rank: 0
#std_ze: [12.64]
#temperature: 0.07
#use_cuda: True
#variables: ['model_108']
#world_size: 1


import os
import sys
import argparse
import numpy as np
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
import netCDF4 as nc

from contextlib import nullcontext
import warnings
warnings.filterwarnings("ignore")


def get_args_parser_icon():

    parser = argparse.ArgumentParser('Extracting features with trained ICON-ATMOS', add_help=False)
    parser.add_argument('--batch_size_per_gpu', default=512, type=int, help='Per-GPU batch-size')
    parser.add_argument('--nb_knn', default=[10, 20, 100, 200], nargs='+', type=int,
        help='Number of NN to use. 20 is usually working the best.')
    parser.add_argument('--temperature', default=0.07, type=float,
        help='Temperature used in the voting coefficient')
    parser.add_argument('--mean_108', default=[266.87], type=float,
        help='mean of ICON 10.8 BT used during normalization') # 266.87 for icon # 270.526 for obs
    parser.add_argument('--std_108', default=[12.64], type=float,
        help='standard deviation of ICON 10.8 BT used during normalization') # 12.64 for icon # 17.901 for obs
    parser.add_argument('--variables', type=str, nargs='+', default=['model_108'],
        help="""time stamp parameters to be used in the model ("--variables 'model_108'" for example)""") # for obs: sample_data
    parser.add_argument('--pretrained_weights', default='/p/scratch/exaww/chatterjee1/nn_obs/all_nc/checkpoint.pth', type=str, help="Path to pretrained weights to evaluate.") #dino_256_trial
    parser.add_argument('--use_cuda', default=True, type=utils_msg108.bool_flag,
        help="Should we store the features on GPU? We recommend setting this to False if you encounter OOM")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--dump_features', default='/p/scratch/exaww/chatterjee1/nn_obs/all_nc/features_icon/', help='Path where to save computed features, empty for no saving')
    parser.add_argument('--load_features', default=None, help="""If the features have
        already been computed, where to find them.""")
    parser.add_argument('--num_workers', default=16, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/p/project/exaww/chatterjee1/dataset/warmworld_datasets/msgobs_108_randcrops_icon.nc', type=str)
    return parser


class CustomDataset_ICON_featureextraction(torch.utils.data.Dataset):
    def __init__(self, nc_file_path, variable_names, transform=None):
        self.nc_file_path = nc_file_path
        self.transform = transform
        self.variable_names = variable_names
        self.data = self._load_data()
        
    def _load_data(self):
        data = {}
        with nc.Dataset(self.nc_file_path, "r") as ds:
            for var_name in self.variable_names:
                data[var_name] = np.array(ds.variables[var_name][:])
        return data

    def __len__(self):
        return len(next(iter(self.data.values())))

    def __getitem__(self, idx):
        sample = {var_name: torch.tensor(self.data[var_name][idx])
                  for var_name in self.variable_names}
        if self.transform:
            sample = self.transform(sample)
        return sample, idx

class ICONAugmentation_featureextraction(object):
    def __init__(self, mean_108, std_108):
        self.normalize = pth_transforms.Normalize(mean=mean_108, std=std_108)

    def __call__(self, sample):
        # Assuming 'image' is a key in the sample dictionary
        image = sample['model_108'] # for icon: model_108, #for obs: sample_data  

        if image.ndim == 2:
            image = image.unsqueeze(0)  # Add a channel dimension

        # Normalize
        crop = self.normalize(image)
    
        return crop 


@torch.no_grad()
def extract_feature_pipeline(args):
    # ============ preparing data ... ============
    transform = ICONAugmentation_featureextraction(
        mean_108=args.mean_108,
        std_108=args.std_108,
    )

    dataset_train = CustomDataset_ICON_featureextraction(
        nc_file_path=args.data_path,
        variable_names=args.variables,
        transform=transform
    )
    sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    print(f"Data loaded with {len(dataset_train)} ICON samples") 

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
        torch.save(train_features.cpu(), os.path.join(args.dump_features, "trainfeat_new_multiyear.pth"))
        torch.save(train_labels.cpu(), os.path.join(args.dump_features, "trainlabels_new_multiyear.pth"))
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
    
    parser = argparse.ArgumentParser('Extracting features with trained ICON-ATMOS', parents=[get_args_parser_icon()])
    args = parser.parse_args()
    utils_msg108.init_distributed_mode(args)
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True
    
    train_features,  train_labels = extract_feature_pipeline(args)

