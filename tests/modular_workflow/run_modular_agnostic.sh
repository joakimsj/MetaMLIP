#!/bin/bash
#SBATCH --partition=standard-g
#SBATCH --account=project_462001547
#SBATCH --time=01:50:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1

set -euo pipefail
export OMP_NUM_THREADS=32
echo $OMP_NUM_THREADS

export MPICH_GPU_SUPPORT_ENABLED=1

## New MACE env ##
export PATH="/project/project_462001097/container_wrapper/mace_env_cueq/bin:\$PATH"

echo "GPU is available/Torch version:"
python3 -c 'import torch; print(torch.cuda.is_available()); print(torch.__version__)'

echo "MACE Version:"
python3 -c "import mace; print(mace.__version__)"

#rm -f MACE_MTD_committee_system.xyz frames_for_DFT_eval.xyz

echo "Running MTD..."
python ald_workflow_surface_agnostic.py --config ald_config.yaml
