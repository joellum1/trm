# Katana environment (login node or PBS job). Large files live on scratch (home is 15 GB).
export SCRATCH=/srv/scratch/$USER
export TRM_ROOT=$HOME/thesis/trm

# gcc 12: system gcc 8.5 is too old to build adam_atan2 against torch
module load gcc/12.2.0 python/3.11.3 cuda/12.8.0
source $SCRATCH/venvs/trm/bin/activate

export HF_HOME=$SCRATCH/trm/hf_cache
export TORCH_EXTENSIONS_DIR=$SCRATCH/trm/torch_extensions   # adam_atan2 JIT build cache
# All Katana GPU types, so one cached build works anywhere (and builds without a GPU)
export TORCH_CUDA_ARCH_LIST="7.0;8.0;8.9;9.0"
export PIP_CACHE_DIR=$SCRATCH/.pip_cache
export WANDB_DIR=$SCRATCH/trm
export WANDB_MODE=${WANDB_MODE:-offline}   # `wandb sync $SCRATCH/trm/wandb/offline-run-*` later
[ -f $HOME/.config/trm/secrets.sh ] && source $HOME/.config/trm/secrets.sh   # HF_TOKEN=... (chmod 600)
export PATH=$HOME/bin:$PATH   # user-local rclone
