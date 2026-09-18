#! /bin/bash
#SBATCH --job-name=multilingual-refusal
#SBATCH -c 8 
#SBATCH --constraint=inet&80gb_vram
#SBATCH -p [REDACTED]
#SBATCH -t 12:00:00 
#SBATCH -G A100:1
#SBATCH --output=./slurm/%x-%j.out     
#SBATCH --error=./slurm/%x-%j.err

CONFIG=<config-path>

# Load HF datasets from cache
export HF_HOME=[REDACTED]
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Environment.
source [REDACTED]
conda activate [REDACTED]
echo "Activated Conda environment: $CONDA_DEFAULT_ENV"

# Paths and nodes.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

# Torch info (uncomment for debugging)
# python --version
# python -m torch.utils.collect_env 2> /dev/null

# Config and sbatch info.
echo "=== Config: $CONFIG ==="
cat "$CONFIG"
echo "========================"

echo "=== SBATCH options ==="
scontrol show job $SLURM_JOB_ID
echo "======================"

# Run pipeline.
python -m pipeline.run_pipeline --config_path "$CONFIG"