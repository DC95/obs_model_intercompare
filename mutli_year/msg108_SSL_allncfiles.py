# CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node=4 msg108_SSL_allncfiles.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import argparse
import h5py
import os
import sys
import numpy as np
from PIL import Image
import xarray as xr
import time
from tqdm import tqdm
import datetime
from pathlib import Path
import math
import datetime

import utils_msg108
#import mlp_nn as mlps
import vision_transformer as vits
from utils_msg108 import get_params_groups
from collections import defaultdict, deque
from vision_transformer import DINOHead
from contextlib import nullcontext
import warnings
warnings.filterwarnings("ignore")

import json
from netCDF4 import Dataset as NC
# MSG-ATMOS: Representing the state of central European atmosphere by exploiting spatial contexts in MSG measurements through self-supervision


def get_args_parser():
    parser = argparse.ArgumentParser('MSG-ATMOS', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'],
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=2048, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils_msg108.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.998, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils_msg108.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.06, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.06, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=70, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils_msg108.bool_flag, default=False, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable.""")
    parser.add_argument('--weight_decay', type=float, default=0.02, help="""Initial value of the
        weight decay.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping.""")
    parser.add_argument('--batch_size_per_gpu', default=70, type=int,
        help='Per-GPU batch-size : number of distinct samples loaded on one GPU.')
    parser.add_argument('--epochs', default=200, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument('--freeze_last_layer', default=20, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")
    
    # data parameters 
    parser.add_argument(
        '--variables',
        type=str, nargs='+', default=['sample_data'],
        help="Variables to read from NetCDF, e.g. --variables sample_data"
    )
    parser.add_argument("--global_size_crops", type=int, default=96, nargs="+",
                        help="crops resolutions (example: [224, 96])")
    parser.add_argument("--local_size_crops", type=int, default=64, nargs="+",
                        help="crops resolutions (example: [224, 96])") 
    
    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=4, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
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
    parser.add_argument('--output_dir', default="/p/scratch/exaww/chatterjee1/nn_obs/all_nc_ps16/", type=str, help='Path to save logs and checkpoints.') ##
    parser.add_argument('--saveckp_freq', default=10, type=int, help='Save checkpoint every x epochs.') 
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=2, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    return parser

try:
    xr.set_options(chunked_array_type="numpy")   # xarray ≥ 2023.06
except Exception:
    pass

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

def train_msg_atmos(args):
    mean_108, std_108 = [270.559], [17.608]
    utils_msg108.init_distributed_mode(args)
    utils_msg108.fix_random_seeds(args.seed)
    cudnn.benchmark = True

    transform = Retroatmos_Augmentation(
        global_crops_scale=args.global_crops_scale,
        local_crops_scale=args.local_crops_scale,
        local_crops_number=args.local_crops_number,
        mean_108=mean_108,
        std_108=std_108,
        args=args,
    )

    dataset = FastNCDataset(args.data_paths, args.variables, transform=transform, keep_open=True)

    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,          # if you see I/O issues, try num_workers=0..2
        pin_memory=True,
        drop_last=True,
        persistent_workers=False,
    )

    print(f"Loaded {len(dataset)} samples from {len(dataset.paths)} NetCDF files.")
    
    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,  # stochastic depth
        )
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size)
        embed_dim = student.embed_dim
    else:
        print(f"Unknow architecture: {args.arch}")
        
    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils_msg108.MultiCropWrapper(student, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ))
    teacher = utils_msg108.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )
    
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils_msg108.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher for MSG-ATMOS are built: they are both {args.arch} network.")
    
     # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        out_dim=args.out_dim,
        ncrops=args.local_crops_number + 2,
        warmup_teacher_temp=args.warmup_teacher_temp,
        teacher_temp=args.teacher_temp,
        warmup_teacher_temp_epochs=args.warmup_teacher_temp_epochs,
        nepochs=args.epochs,
        student_temp=0.12,                # e.g., slightly warmer student
        center_momentum=0.9,              # fallback if no schedule provided
        center_mom_boundaries=[30, 100],  # epochs where value changes
        center_mom_values=[0.85, 0.90, 0.95]
    ).cuda()
    
    # ============ preparing optimizer ... ============
    params_groups = utils_msg108.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()
        
    # ============ init schedulers ... ============
    lr_schedule = utils_msg108.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils_msg108.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils_msg108.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils_msg108.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")
    
    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils_msg108.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]
    
    start_time = time.time()
    print("Starting MSG-ATMOS training !")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)


        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils_msg108.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils_msg108.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils_msg108.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
            
                 
                 
def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch,
                    fp16_scaler, args):
    metric_logger = utils_msg108.MetricLogger(delimiter="  ")
    header = f"Epoch: [{epoch}/{args.epochs}]"

    # modes
    student.train()
    teacher.eval()

    use_amp = fp16_scaler is not None
    amp_ctx = torch.cuda.amp.autocast if use_amp else nullcontext

    for it, (images, idx) in enumerate(metric_logger.log_every(data_loader, 10, header)):

        if it == 0 and utils_msg108.is_main_process():
            print("num crops:", len(images), "crop shapes:", [tuple(t.shape) for t in images])

        # move images to GPU
        images = [im.cuda(non_blocking=True) for im in images]
        # idx is unused; skip moving it unless you need it
        # idx = idx.cuda(non_blocking=True)

        # schedules
        global_step = len(data_loader) * epoch + it
        for i, pg in enumerate(optimizer.param_groups):
            pg["lr"] = lr_schedule[global_step]
            if i == 0:
                pg["weight_decay"] = wd_schedule[global_step]

        # forward + loss
        with amp_ctx():
            with torch.no_grad():
                teacher_output = teacher(images[:2])  # 2 globals
            student_output = student(images)
            loss = dino_loss(student_output, teacher_output, epoch)

        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping training", force=True)
            sys.exit(1)

        # backward
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)
                _ = utils_msg108.clip_gradients(student, args.clip_grad)
            utils_msg108.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        else:
            loss.backward()
            if args.clip_grad:
                _ = utils_msg108.clip_gradients(student, args.clip_grad)
            utils_msg108.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            optimizer.step()

        # EMA update for teacher (no grad)
        with torch.no_grad():
            m = momentum_schedule[global_step]
            for p_q, p_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                p_k.data.mul_(m).add_((1.0 - m) * p_q.detach().data)

        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item(),
                             lr=optimizer.param_groups[0]["lr"],
                             wd=optimizer.param_groups[0]["weight_decay"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
   
        
class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.14,
                 center_momentum=0.9,   # float OR tuple/list for schedule config
                 center_mom_boundaries=None,  # e.g., [30, 100]
                 center_mom_values=None      # e.g., [0.85, 0.90, 0.95]
                 ):
        super().__init__()
        self.student_temp = student_temp
        self.ncrops = ncrops
        self.nepochs = nepochs
        self.register_buffer("center", torch.zeros(1, out_dim))

        # teacher temperature schedule (unchanged)
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

        # ---- NEW: center momentum schedule ----
        self.center_mom_schedule = self._build_center_mom_schedule(
            nepochs=nepochs,
            center_momentum=center_momentum,
            boundaries=center_mom_boundaries,
            values=center_mom_values
        )

    def _build_center_mom_schedule(self, nepochs, center_momentum,
                                   boundaries=None, values=None):
        """
        Returns a numpy array of length `nepochs` with momentum per epoch.
        - If `boundaries` and `values` are provided: piecewise-constant schedule.
        - Else: constant schedule using `center_momentum`.
        """
        if boundaries is not None and values is not None:
            assert len(values) == len(boundaries) + 1, \
                "values must have exactly one more element than boundaries"
            sched = np.empty(nepochs, dtype=np.float32)
            start = 0
            for b, v in zip(list(boundaries) + [nepochs], values):
                end = b if isinstance(b, int) else int(b)  # ensure int
                sched[start:end] = v
                start = end
            return sched
        else:
            return np.full(nepochs, float(center_momentum), dtype=np.float32)

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        # student softmax temp
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering + sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)  # 2 global crops

        total_loss, n_loss_terms = 0.0, 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms

        self.update_center(teacher_output, epoch)   # <--- pass epoch here
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output, epoch):
        """
        Update center with epoch-conditioned momentum.
        """
        # batch_center across GPUs
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_size_total = teacher_output.shape[0] * dist.get_world_size()
        batch_center = batch_center / batch_size_total

        m = float(self.center_mom_schedule[min(epoch, self.nepochs - 1)])
        self.center = self.center * m + batch_center * (1.0 - m)

class Retroatmos_Augmentation(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number, mean_108, std_108, args):
        self.global_transfo1 = utils_msg108.PartialScaledRandomCrop(args.global_size_crops,args.global_size_crops, global_crops_scale)
        self.global_transfo2 = utils_msg108.PartialScaledRandomCrop(args.global_size_crops,args.global_size_crops, global_crops_scale)
        self.local_crops_number = local_crops_number
        self.local_transfo = utils_msg108.PartialScaledRandomCrop(args.local_size_crops,args.local_size_crops, local_crops_scale)
        self.normalize = transforms.Normalize(mean=mean_108, std=std_108)

    def __call__(self, sample):
        
        data = sample['sample_data']
        
        if not isinstance(data, torch.Tensor): #or not isinstance(vm, torch.Tensor) or not isinstance(tb, torch.Tensor):
            raise TypeError("Expected 'data' to be tensors.") # and 'vm' and 'tb' 
        
        # Ensure both tensors have a channel dimension, if necessary
        if data.dim() == 2:
            data = data.unsqueeze(0)  
        
        # Concatenate along the channel dimension (dim=0)
        image = data

        # Apply the global transformations
        crop1 = self.global_transfo1(image)
        crop2 = self.global_transfo2(image)
        
        # Normalize
        crop1 = self.normalize(crop1)
        crop2 = self.normalize(crop2)
        
        # Prepare list of crops
        crops = [crop1, crop2]

        # Apply local transformations
        for _ in range(self.local_crops_number):
            crop = self.local_transfo(image)
            crop = self.normalize(crop)
            crops.append(crop)
        
        return crops
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser('MSG-ATMOS', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_msg_atmos(args)
