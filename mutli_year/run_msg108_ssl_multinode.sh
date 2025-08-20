#!/bin/bash -x
#SBATCH --account=exaww
#SBATCH --partition=booster
#SBATCH --nodes=3                 # <<< set the number of nodes you want
#SBATCH --gres=gpu:4              # Booster has 4 GPUs per node
#SBATCH --ntasks-per-node=1       # one launcher per node; torchrun spawns local ranks
#SBATCH --cpus-per-task=12        # tune for your dataloader
#SBATCH --time=24:00:00
#SBATCH -D /p/project1/exaww/chatterjee1/mcspss_continuous
#SBATCH -o /p/project1/exaww/chatterjee1/mcspss_continuous/logs/%x_%j.out
#SBATCH -e /p/project1/exaww/chatterjee1/mcspss_continuous/logs/%x_%j.err

set -euo pipefail
mkdir -p /p/project1/exaww/chatterjee1/mcspss_continuous/logs

# --- Env ---
source /p/project1/exaww/chatterjee1/juwels_env_2024/activate.sh

# Sanity breadcrumbs (very useful when things go sideways)
echo "JobID: $SLURM_JOB_ID"; echo "Nodes: $SLURM_NODELIST"; date
python -V; which python || true; nvidia-smi || true

# --- NCCL / threading knobs (safe defaults) ---
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=ib0      # change if your IB iface differs

# --- torchrun rendezvous derived from Slurm ---
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_PORT=29500                  # change if this port is busy
NNODES=${SLURM_NNODES}
GPUS_PER_NODE=4

# Launch one torchrun per node; each spawns 4 local ranks
srun --ntasks=$NNODES --ntasks-per-node=1 --cpu-bind=none \
  torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    /p/project1/exaww/chatterjee1/mcspss_continuous/msg108_SSL_allncfiles.py \
    --dist_url env://