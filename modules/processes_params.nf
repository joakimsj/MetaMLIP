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

  publishDir "results/${params.run_name}/${run_label}/runMACE", mode: 'copy'

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
    export MPICH_GPU_SUPPORT_ENABLED=1
    ## Environment path ##
    export PATH="${params.env_path.mace}/bin:\$PATH"

    echo "GPU is available/Torch version:"
    python3 -c 'import torch; print(torch.cuda.is_available()); print(torch.__version__)'

    echo "MACE Version:"
    python3 -c "import mace; print(mace.__version__)"

    echo "Model files: ${model_paths_string}"

    #############################################
    # ADAPTIVE SAMPLING LOOP: REPEAT UNTIL ≥ 20
    #############################################

    while true; do

        echo "Cleaning previous MTD output..."
        rm -f MACE_MTD_committee_system.xyz # frames_for_DFT_eval.xyz

        echo "Running MTD..."
        python ${propagatorMTD} \
            --input_file ${initialFrame} \
            --model_paths ${model_paths_string} \
            --timestep ${params.mtd.timestep} \
            --temperature ${params.mtd.temperature} \
            --kappa ${params.mtd.kappa} \
            --wall_pos ${params.mtd.wall_pos} \
            --pace ${params.mtd.pace} \
            --height ${params.mtd.height} \
            --z_threshold ${params.mtd.z_threshold} \
            --sigma1 ${params.mtd.sigma1} \
            --sigma2 ${params.mtd.sigma2} \
            --biasfactor ${params.mtd.biasfactor} \
            --nsteps ${params.mtd.nsteps} \
            --variance_limit ${params.mtd.variance_limit} \
            --force_variance_limit ${params.mtd.force_variance_limit} \
            --interval ${params.mtd.interval} \
            --stride ${params.mtd.stride} \
            --c1_threshold ${params.mtd.c1_threshold} \
            --c2_threshold ${params.mtd.c2_threshold}
        echo "Done!"
        
        echo "Running descriptor filter..."
        set +e
        python ${descriptorFilter} \
            --new frames_for_DFT_eval.xyz \
            --reference ${growingDataset} \
            --threshold ${params.filtering.descriptor_threshold} \
            --max_structures ${params.filtering.max_structures}

        status=\$?
        set -e 
        
        # ----------------------------------------
        #  Exit code interpretation:
        #    0  -> Enough structures collected
        #    10 -> <20 structures -> repeat MTD
        #  other -> fatal error
        # ----------------------------------------

        if [[ \$status -eq 10 ]]; then
            echo "Fewer than 20 total structures — repeating metadynamics..."
            continue
        elif [[ \$status -eq 0 ]]; then
            echo "Enough structures collected — proceeding!"
            break
        else
            echo "Descriptor filter error (exit code \$status). Aborting."
            exit \$status
        fi

    done

    echo "Filtering done!"
    """
}


process runMACE_no_adaptive_sampling {
  label 'gpu_mace_run'

  input:
    path propagatorMTD
    path descriptorFilter 
    path initialFrame
    path growingDataset
    val model_files 
    val run_label

  publishDir "results/${params.run_name}/${run_label}/runMACE_no_adaptive_sampling", mode: 'copy'
  
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

    export OMP_NUM_THREADS=${params.omp_threads_gpu}
    echo \$OMP_NUM_THREADS

    export MPICH_GPU_SUPPORT_ENABLED=1

    ## Environment path ##
    export PATH="${params.env_path.mace}/bin:\$PATH"

    echo "GPU is available/Torch version:"
    python3 -c 'import torch; print(torch.cuda.is_available()); print(torch.__version__)'

    echo "MACE Version:"
    python3 -c "import mace; print(mace.__version__)"

    rm -f MACE_MTD_committee_system.xyz frames_for_DFT_eval.xyz
    
    echo "Model files: ${model_paths_string}"
    
    echo "Running MTD..."
    python ${propagatorMTD} \
        --input_file ${initialFrame} \
        --model_paths ${model_paths_string} \
        --timestep ${params.mtd.timestep} \
        --temperature ${params.mtd.temperature} \
        --kappa ${params.mtd.kappa} \
        --wall_pos ${params.mtd.wall_pos} \
        --pace ${params.mtd.pace} \
        --height ${params.mtd.height} \
        --z_threshold ${params.mtd.z_threshold} \
        --sigma1 ${params.mtd.sigma1} \
        --sigma2 ${params.mtd.sigma2} \
        --biasfactor ${params.mtd.biasfactor} \
        --nsteps ${params.mtd.nsteps} \
        --variance_limit ${params.mtd.variance_limit} \
        --force_variance_limit ${params.mtd.force_variance_limit} \
        --interval ${params.mtd.interval} \
        --stride ${params.mtd.stride} \
        --c1_threshold ${params.mtd.c1_threshold} \
        --c2_threshold ${params.mtd.c2_threshold}
    echo "Done!"

    echo "Filtering structures based on MACE-descriptors..."
    python ${descriptorFilter} \
        --new frames_for_DFT_eval.xyz \
        --reference ${growingDataset} \
        --threshold ${params.filtering.descriptor_threshold} \
        --max_structures ${params.filtering.max_structures}
    echo "Filtering done!"
    """
}

process runMACE_MTD {
  label 'gpu_mace_run'

  input:
    path propagatorMTD
    path descriptorFilter 
    path initialFrame
    path growingDataset
    val model_files 
    val run_label

  publishDir "results/${params.run_name}/${run_label}/runMACE_MTD", mode: 'copy'
  
  output:
    path "MACE_MTD_committee_system.xyz", emit: mtd_trajectory
    path "*.xyz"
    path "*.png"
    path "COLVAR", emit: colvar_file
    path "HILLS"

  script:
    def model_paths_string = model_files.join(' ')
    """
    set -euo pipefail

    export OMP_NUM_THREADS=${params.omp_threads_gpu}
    echo \$OMP_NUM_THREADS

    export MPICH_GPU_SUPPORT_ENABLED=1

    ## Environment path ##
    export PATH="${params.env_path.mace}/bin:\$PATH"

    echo "GPU is available/Torch version:"
    python3 -c 'import torch; print(torch.cuda.is_available()); print(torch.__version__)'

    echo "MACE Version:"
    python3 -c "import mace; print(mace.__version__)"

    rm -f MACE_MTD_committee_system.xyz frames_for_DFT_eval.xyz
    
    echo "Model files: ${model_paths_string}"
    
    echo "Running MTD..."
    python ${propagatorMTD} \
        --input_file ${initialFrame} \
        --model_paths ${model_paths_string} \
        --timestep ${params.mtd.timestep} \
        --temperature ${params.mtd.temperature} \
        --kappa ${params.mtd.kappa} \
        --wall_pos ${params.mtd.wall_pos} \
        --pace ${params.mtd.pace} \
        --height ${params.mtd.height} \
        --z_threshold ${params.mtd.z_threshold} \
        --sigma1 ${params.mtd.sigma1} \
        --sigma2 ${params.mtd.sigma2} \
        --biasfactor ${params.mtd.biasfactor} \
        --nsteps ${params.mtd.nsteps} \
        --variance_limit ${params.mtd.variance_limit} \
        --force_variance_limit ${params.mtd.force_variance_limit} \
        --interval ${params.mtd.interval} \
        --stride ${params.mtd.stride} \
        --c1_threshold ${params.mtd.c1_threshold} \
        --c2_threshold ${params.mtd.c2_threshold}
    echo "Done!"
    """
}

process runMACE_NEB_sequential {
  label 'gpu_mace_run'

  input:
    path committeeNEB
    path descriptorFilter 
    path reaction_window
    path growingDataset
    val model_files 
    val run_label

  publishDir "results/${params.run_name}/${run_label}/runMACE_NEB_sequential", mode: 'copy'
  
  output:
    path "*_neb_dft_harvest.xyz", emit: neb_uncertain_frames
    path "*.xyz"

  script:
    def model_paths_string = model_files.join(' ')
    """
    set -euo pipefail

    export OMP_NUM_THREADS=${params.omp_threads_gpu}
    echo \$OMP_NUM_THREADS

    export MPICH_GPU_SUPPORT_ENABLED=1

    ## Environment path ##
    export PATH="${params.env_path.mace}/bin:\$PATH"

    echo "GPU is available/Torch version:"
    python3 -c 'import torch; print(torch.cuda.is_available()); print(torch.__version__)'

    echo "MACE Version:"
    python3 -c "import mace; print(mace.__version__)"

    echo "Model files: ${model_paths_string}"
    
    for window in reaction_*_window.extxyz; do
        echo "Running NEB on \$window"

        python ${committeeNEB} \
        --initial_file "\$window" \
        --model_paths ${model_paths_string} \
        --z_threshold ${params.neb.z_threshold} \
        --n_images ${params.neb.n_images} \
        --n_images_eval ${params.neb.n_images_eval}

    done
    
    echo "Done!"
    """
}

process runMACE_NEB {
  label 'gpu_mace_run'
  maxForks 4
  
  input:
    path committeeNEB
    each reaction_window
    path growingDataset 
    val model_files 
    val run_label

  publishDir "results/${params.run_name}/${run_label}/runMACE_NEB", mode: 'copy'
  
  output:
    path "*_neb_dft_harvest.xyz", emit: neb_uncertain_frames
    path "*.xyz"

  script:
    def model_paths_string = model_files.join(' ')
    """
    set -u

    export OMP_NUM_THREADS=${params.omp_threads_gpu}
    export MPICH_GPU_SUPPORT_ENABLED=1
    
    ## Environment path ##
    export PATH="${params.env_path.mace}/bin:\$PATH"

    #rid=\$(basename "${reaction_window}" .extxyz)
    #mkdir -p "neb_\$rid"
    #cd "neb_\$rid"

    echo "Running NEB on ${reaction_window}"

    python ${committeeNEB} \
        --initial_file "${reaction_window}" \
        --model_paths ${model_paths_string} \
        --z_threshold ${params.neb.z_threshold}  \
        --n_images ${params.neb.n_images}  \
        --n_images_eval ${params.neb.n_images_eval} || echo "NEB failed for ${reaction_window}"

    echo "Done: ${reaction_window}"
    """
}

process extractReact {
  label 'local'
  
  input:
    path NEB_extractor
    path mtd_trajectory
    path colvar_file
    val run_label
    
  output:
    path "reaction_*_window.extxyz", emit: reaction_windows
    path "reaction_indices.txt"
    path "reaction_bonds.log"

  publishDir "results/${params.run_name}/${run_label}/extractReact", mode: 'copy'

  script:
    """
    export MPICH_GPU_SUPPORT_ENABLED=1

    ## Environment path ##
    export PATH="${params.env_path.mace}/bin:\$PATH"   
    
    echo "Extracting reactions..."

    python ${NEB_extractor} \
        --input_file ${mtd_trajectory} \
        --colvar ${colvar_file} \
        --output "reaction_frames.extxyz" \
        --indices-output "reaction_indices.txt" \
        --delta ${params.extract.delta} \
        --cv ${params.extract.cv} \
        --padding ${params.extract.padding} \
        --persist ${params.extract.persist} \
        --pre-frames ${params.extract.pre_frames} \
        --post-frames ${params.extract.post_frames} \
        --form-scale ${params.extract.form_scale} \
        --break-scale ${params.extract.break_scale} \
        --min_spacing ${params.extract.min_spacing} \
        --zmin ${params.extract.zmin} \
        --max-reactions ${params.extract.max_reactions} \
        --debug
    
    echo "Done!"
    """
}

process collectNEBs {
  label 'gpu_mace_run'
  maxForks 1
  
  input:
    path descriptorFilter
    path neb_uncertain_frames
    path growingDataset
    val run_label
    
  output:
    path "frames_for_DFT_eval_filtered.xyz", emit: mace_frames
    path "growing_dataset.xyz", emit: updated_dataset
    path "growing_dataset_*.xyz", emit: backup_dataset

  publishDir "results/${params.run_name}/${run_label}/collectNEBs", mode: 'copy'

  script:
    """
    ## Environment path ##
    export PATH="${params.env_path.mace}/bin:\$PATH"
    
    echo "NEB windows received:"
    ls -lh ${neb_uncertain_frames}
    
    echo "Collecting NEB data for filtering..."
    cat ${neb_uncertain_frames} >> frames_for_DFT_eval.xyz
    echo "Done collecting"
    
    
    echo "Filtering structures based on MACE-descriptors..."
   
    python ${descriptorFilter} \
        --new frames_for_DFT_eval.xyz \
        --reference ${growingDataset} \
        --threshold ${params.filtering.descriptor_threshold} \
        --max_structures ${params.filtering.max_structures}
    
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
    val run_label

  output:
    path "cp2k_farmed_dataset.xyz", emit: new_data
    path "*.xyz"
    path "run*"
    path "*.out"

  publishDir "results/${params.run_name}/${run_label}/calcREF", mode: 'copy'

  script:
    """
    set -euo pipefail
    
    export OMP_PLACES=cores
    export OMP_PROC_BIND=close
    export OMP_NUM_THREADS=${params.cp2k.omp_threads}
    ulimit -s unlimited
    
    ## Environment path ##
    export PATH="${params.env_path.mace}/bin:\$PATH"

    echo "Loading in CP2K modules.."
    module use /appl/local/csc/modulefiles
    module load cp2k/2025.1
    echo "Modules loaded!"

    echo "Sowing seeds..."
    python ${prepare_cp2k_input}
    echo "Seeds sown for cp2k farming!"

    echo "Harvest time!"
    if ! srun cp2k.psmp farming_driver.inp > farming.out 2> farming.err; then
        echo "WARNING: CP2K farming failed for some inputs (see farming.err)" >&2
    fi
    echo "CP2K calcs finished (some may have failed)."
    
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
    val run_label
    
  output:
    path "growing_dataset.xyz", emit: updated_dataset
    path "growing_dataset_*.xyz", emit: backup_dataset
    //path "growing_retrain_dataset.xyz", emit: persistent_dataset

  publishDir "results/${params.run_name}/${run_label}/updateDataset", mode: 'copy'

  script:
    """
    echo "Updating dataset..."
    
    # Create a temporary file to avoid overwriting existing_dataset before reading
    tmp_file=\$(mktemp)

    # Concatenate existing dataset and new data into the temp file
    cat ${existing_dataset} ${new_data} > \${tmp_file}

    # Move temp file to growing_dataset.xyz
    mv \${tmp_file} growing_dataset.xyz

    # Backup with timestamp
    timestamp=\$(date +%Y%m%d_%H%M%S)
    backup_name="growing_dataset_\${timestamp}.xyz"
    cp growing_dataset.xyz \${backup_name}

    mkdir -p growing_dataset/backup
    cp \${backup_name} growing_dataset/backup/
    echo "Dataset updated and backed up as \${backup_name}"
    """
}


process reTrainMACE {
  label 'gpu_mace_train'

  input:
    path cp2k_dataset
    path foundation_model
    each seed   // fan-out over seeds
    val run_label
    
  output:
    path "MACE_model_seed_*.model", emit: trained_models
    path "results"
    path "logs"
    path "checkpoints"

  publishDir "results/${params.run_name}/${run_label}/reTrainMACE/seed_${seed}", mode: 'copy'
  
  script:
    """
    set -euo pipefail

    # Use centralized parameter values
    export OMP_NUM_THREADS=${params.retrain.omp_threads}
    export MPICH_GPU_SUPPORT_ENABLED=1
    export PATH="${params.env_path.mace}/bin:\$PATH"

    echo "Running MACE training with seed $seed"

    mace_run_train \
      --name="MACE_model_seed_${seed}" \
      --train_file="${cp2k_dataset}" \
      --valid_fraction=${params.retrain.valid_fraction} \
      --config_type_weights=' ' \
      --atomic_numbers="${params.retrain.atomic_numbers}" \
      --E0s='${params.retrain.E0s}' \
      --model="MACE" \
      --foundation_model="${foundation_model}" \
      --pt_train_file="${params.retrain.pt_train_file}" \
      --num_samples_pt=${params.retrain.num_samples_pt} \
      --batch_size=${params.retrain.batch_size} \
      --multiheads_finetuning=${params.retrain.multiheads_finetuning} \
      --energy_key="${params.retrain.energy_key}" \
      --forces_key="${params.retrain.forces_key}" \
      --hidden_irreps="${params.retrain.hidden_irreps}" \
      --r_max=${params.retrain.r_max} \
      --foundation_filter_elements=${params.retrain.foundation_filter_elements} \
      --filter_type_pt="${params.retrain.filter_type_pt}" \
      --forces_weight=${params.retrain.forces_weight} \
      --energy_weight=${params.retrain.energy_weight} \
      --stress_weight=${params.retrain.stress_weight} \
      --max_num_epochs=${params.retrain.max_num_epochs} \
      --restart_latest \
      --device=${params.retrain.device} \
      --swa \
      --swa_energy_weight=${params.retrain.swa_energy_weight} \
      --swa_forces_weight=${params.retrain.swa_forces_weight} \
      --swa_stress_weight=${params.retrain.swa_stress_weight} \
      --seed=${seed}
    """
}

process reTrainMACE_naive {
  label 'gpu_mace_train'

  input:
    path cp2k_dataset
    path foundation_model
    each seed   // fan-out over seeds
    val run_label
    
  output:
    path "MACE_model_seed_*.model", emit: trained_models
    path "results"
    path "logs"
    path "checkpoints"

  publishDir "results/${params.run_name}/${run_label}/reTrainMACE_naive/seed_${seed}", mode: 'copy'

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
      --valid_fraction=${params.retrain.valid_fraction} \
      --config_type_weights=' ' \
      --atomic_numbers="${params.retrain.atomic_numbers}" \
      --E0s='${params.retrain.E0s}' \
      --model="MACE" \
      --foundation_model="${foundation_model}" \
      --multiheads_finetuning=False \
      --pt_train_file="mp" \
      --num_samples_pt=1000 \
      --energy_key="REF_energy" \
      --forces_key="REF_forces" \
      --hidden_irreps="128x0e + 128x1o" \
      --r_max=6.0 \
      --foundation_filter_elements=True \
      --filter_type_pt="combinations" \
      --forces_weight=10 \
      --energy_weight=1 \
      --stress_weight=0 \
      --max_num_epochs=50 \
      --restart_latest \
      --device=cuda \
      --swa \
      --swa_energy_weight=10.0 \
      --swa_forces_weight=100 \
      --swa_stress_weight=0 \
      --seed=${seed}
    """
}

process reTrainMACE_recursive {
  label 'gpu_mace_train'

  input:
    path cp2k_dataset
    each foundation_model
    path existing_dataset
    val seed
    val run_label
    
  output:
    path "*.model", emit: trained_models
    path "results"
    path "logs"
    path "checkpoints"

  publishDir "results/reTrainMACE_recursive/${run_label}/${foundation_model.baseName}", mode: 'copy'

  script:
    """
    set -euo pipefail

    export OMP_NUM_THREADS=32
    export MPICH_GPU_SUPPORT_ENABLED=1
    export PATH="/project/project_462000838/container_wrapper/mace_env_cueq/bin:\$PATH"

    echo "Running MACE training for foundation model: ${foundation_model.getName()} with seed ${seed}"

    mace_run_train \
      --name="${foundation_model.baseName}_${run_label}" \
      --train_file="${cp2k_dataset}" \
      --atomic_numbers="[1, 6, 7, 8, 14]" \
      --E0s='{1:-13.55946263, 6:-157.53735191, 7:-265.91593046, 8:-431.59585675, 14:-102.46189747}' \
      --foundation_model="${foundation_model}" \
      --pt_train_file="${existing_dataset}" \
      --num_samples_pt=50 \
      --multiheads_finetuning=True \
      --energy_key="REF_energy" \
      --forces_key="REF_forces" \
      --hidden_irreps="128x0e + 128x1o" \
      --r_max=6.0 \
      --batch_size=2 \
      --foundation_filter_elements=True \
      --filter_type_pt="combinations" \
      --subselect_pt fps \
      --forces_weight=10 \
      --energy_weight=1 \
      --stress_weight=0 \
      --compute_stress=False \
      --max_num_epochs=50 \
      --restart_latest \
      --device=cuda \
      --swa \
      --swa_energy_weight=10.0 \
      --swa_forces_weight=100 \
      --swa_stress_weight=0 \
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

