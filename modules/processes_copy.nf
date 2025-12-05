#!/usr/bin/env nextflow

process runMACE {
  label 'gpu_mace_run'

  input:
    path propagatorMTD
    path descriptorFilter 
    path initialFrame
    path growingDataset
    val model_files 
    val run_label

  publishDir "results/runMACE/${run_label}", mode: 'copy'
  
  output:
    path "frames_for_DFT_eval_filtered.xyz", emit: mace_frames
    path "*.xyz"
    path "*.png"
    path "COLVAR"
    path "HILLS"

  script:
    def model_paths_string = model_files instanceof List ? model_files.join(' ') : model_files
    """
    set -euo pipefail

    export OMP_NUM_THREADS=32
    echo \$OMP_NUM_THREADS

    export MPICH_GPU_SUPPORT_ENABLED=1

    ## New MACE env ##
    export PATH="/project/project_462000838/container_wrapper/mace_env_cueq/bin:\$PATH"

    echo "GPU is available/Torch version:"
    python3 -c 'import torch; print(torch.cuda.is_available()); print(torch.__version__)'

    echo "MACE Version:"
    python3 -c "import mace; print(mace.__version__)"

    rm -f MACE_MTD_committee_system.xyz frames_for_DFT_eval.xyz
    
    echo "Model files: ${model_paths_string}"
    
    echo "Running MTD..."
    python ${propagatorMTD} --input_file ${initialFrame} --model_paths ${model_paths_string} \
      --timestep 1.0 \
      --temperature 400 \
      --pace 400 \
      --height 4.0 \
      --sigma1 0.1 \
      --sigma2 0.2 \
      --biasfactor 5 \
      --nsteps 10000 \
      --variance_limit 0.0020 \
      --interval 5 \
      --stride 10 \
      --c1_threshold 0.8 \
      --c2_threshold 2.5
    echo "Done!"

    echo "Filtering structures based on MACE-descriptors..."
    python ${descriptorFilter} --new frames_for_DFT_eval.xyz --reference ${growingDataset} --threshold 5 --max_structures 50
    echo "Filtering done!"
    """
}

process calcREF {
  label 'cp2k_farming'

  input:
    path prepare_cp2k_input
    path parse_cp2k_output
    path template
    path frames_from_MTD

  output:
    path "cp2k_farmed_dataset.xyz", emit: new_data
    path "*.xyz"
    path "run*"
    path "*.out"

  script:
    """
    set -euo pipefail
    
    export OMP_PLACES=cores
    export OMP_PROC_BIND=close
    export OMP_NUM_THREADS=2
    ulimit -s unlimited
    
    export PATH="/project/project_462000838/container_wrapper/mace_env_cueq/bin:$PATH"

    echo "Loading in CP2K modules.."
    module use /appl/local/csc/modulefiles
    module load cp2k/2024.3
    echo "Modules loaded!"

    echo "Sowing seeds..."
    python ${prepare_cp2k_input}
    echo "Seeds sown for cp2k farming!"

    echo "Harvest time!"
    srun cp2k.psmp farming_driver.inp > farming.out
    echo "CP2K calcs harvested!"
    
    echo "Parsing harvest, preparing extxyz/xyz files for the winter..."
    python ${parse_cp2k_output}
    echo "Harvest has been parsed!"
    """
}

process updateDataset {
  label 'local'
  
  input:
    path new_data
    path existing_dataset

  output:
    path "growing_dataset.xyz", emit: updated_dataset
    path "growing_dataset_*.xyz", emit: backup_dataset

  script:
    """
    echo "Updating dataset..."

    cat ${existing_dataset} ${new_data} > growing_dataset.xyz

    timestamp=\$(date +%Y%m%d_%H%M%S)
    
    backup_name="growing_dataset_\${timestamp}.xyz"
    cp growing_dataset.xyz \${backup_name}

    mkdir -p growing_dataset/backup
    cp \${backup_name} growing_dataset/backup/

    cp growing_dataset.xyz growing_dataset/growing_retrain_dataset.xyz
    """
}


process reTrainMACE {
  label 'gpu_mace_train'

  input:
    path cp2k_dataset
    path foundation_model
    val seed
  
  output:
    path "*.model", emit: trained_models
    path "results"
    path "logs"
    path "checkpoints"


  publishDir "results/reTrainMACE/seed_${seed}", mode: 'copy'

  script:
    """
    set -euo pipefail

    export OMP_NUM_THREADS=32
    export MPICH_GPU_SUPPORT_ENABLED=1
    export PATH="/project/project_462000838/container_wrapper/mace_env_cueq/bin:\$PATH"

    echo "Running MACE training with seed $seed"

    mace_run_train \
      --name="MACE_model_seed_${seed}" \
      --train_file="${cp2k_dataset}" \
      --valid_fraction=0.05 \
      --config_type_weights=' ' \
      --atomic_numbers="[1, 6, 7, 8, 14]" \
      --E0s='{1:-13.55946263, 6:-157.53735191, 7:-265.91593046, 8:-431.59585675, 14:-102.46189747}' \
      --model="MACE" \
      --foundation_model="${foundation_model}" \
      --pt_train_file="mp" \
      --num_samples_pt=1000 \
      --multiheads_finetuning=True \
      --energy_key="REF_energy" \
      --forces_key="REF_forces" \
      --hidden_irreps="128x0e + 128x1o" \
      --r_max=6.0 \
      --foundation_filter_elements=True \
      --filter_type_pt="combinations" \
      --forces_weight=100 \
      --energy_weight=1 \
      --stress_weight=0 \
      --max_num_epochs=40 \
      --restart_latest \
      --device=cuda \
      --swa \
      --swa_energy_weight 10.0 \
      --swa_forces_weight=100 \
      --seed=${seed}
    """
}

process runMACE_retrained{
  label 'gpu_mace_run'

  input:
    path propagatorMTD
    path descriptorFilter 
    path initialFrame
    path growingDataset
    val model_files 
    val run_label

  publishDir "results/runMACE/${run_label}", mode: 'copy'
  
  output:
    path "frames_for_DFT_eval_filtered.xyz", emit: mace_frames
    path "*.xyz"
    path "*.png"
    path "COLVAR"
    path "HILLS"

  script:
      def model_paths_string = model_files.join(' ')
      """
      set -euo pipefail

      export OMP_NUM_THREADS=32
      echo \$OMP_NUM_THREADS

      export MPICH_GPU_SUPPORT_ENABLED=1

      ## New MACE env ##
      export PATH="/project/project_462000838/container_wrapper/mace_env_cueq/bin:\$PATH"

      echo "GPU is available/Torch version:"
      python3 -c 'import torch; print(torch.cuda.is_available()); print(torch.__version__)'

      echo "MACE Version:"
      python3 -c "import mace; print(mace.__version__)"

      rm -f MACE_MTD_committee_system.xyz frames_for_DFT_eval.xyz

      echo "Model files: ${model_paths_string}"

      echo "Running MTD..."
      python ${propagatorMTD} --input ${initialFrame} --model_paths ${model_paths_string} \
        --timestep 1.0 \
        --temperature 400 \
        --pace 400 \
        --height 4.0 \
        --sigma1 0.1 \
        --sigma2 0.2 \
        --biasfactor 5 \
        --nsteps 2500 \
        --variance_limit 0.0020 \
        --interval 5 \
        --stride 10 \
        --c1_threshold 0.8 \
        --c2_threshold 2.5        
      echo "Done!"

      echo "Filtering structures based on MACE-descriptors..."
      python ${descriptorFilter} --new frames_for_DFT_eval.xyz --reference ${growingDataset} --threshold 5 --max_structures 50
      echo "Filtering done!"
      """
}

