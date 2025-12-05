#!/bin/bash
#SBATCH --partition=standard-g
#SBATCH --account=project_462000838
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1

export OMP_NUM_THREADS=32
echo $OMP_NUM_THREADS

export MPICH_GPU_SUPPORT_ENABLED=1

module use /appl/local/csc/modulefiles
module load cp2k/2024.3-gpu
#module load LAMMPS/mace-flare-18Feb2025-cpeAMD-24.03-rocm-libtorch-mace-flare-KOKKOS
#module list

## New MACE env ##
export PATH="/project/project_462000838/container_wrapper/mace_env_cueq/bin:$PATH"

## Old MACE env, works, but old mace and no cuequivariance acceleration ##
#export PATH="/project/project_462000290/jestilj1/container_wrapper/flare_mace_ase/bin:$PATH"

echo "GPU is available/Torch version:"
python3 -c 'import torch; print(torch.cuda.is_available()); print(torch.__version__)'

echo "MACE Version:"
python3 -c "import mace; print(mace.__version__)"

mace_run_train \
--name="MACE_model_finetuned_lumi_test" \
--train_file="cp2k_results_both_CVs.extxyz" \
--valid_fraction=0.10 \
--config_type_weights=' ' \
--atomic_numbers="[1, 6, 7, 8, 14]" \
--E0s='{1:-13.55946263, 6:-157.53735191, 7:-265.91593046, 8:-431.59585675, 14:-102.46189747}' \
--model="MACE" \
--foundation_model="mace-mpa-0-medium.model" \
--pt_train_file="mp" \
--num_samples_pt=1000 \
--multiheads_finetuning=True \
--energy_key="REF_energy" \
--forces_key="REF_forces" \
--hidden_irreps="128x0e + 128x1o" \
--r_max=5.0 \
--foundation_filter_elements=True \
--filter_type_pt="combinations" \
--forces_weight=10 \
--energy_weight=1 \
--stress_weight=0 \
--batch_size=4 \
--max_num_epochs=5 \
--swa \
--start_swa=64 \
--ema \
--ema_decay=0.99 \
--amsgrad \
--restart_latest \
--device=cuda \
--seed=789 \

