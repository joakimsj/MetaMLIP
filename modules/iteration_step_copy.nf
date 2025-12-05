nextflow.enable.dsl=2

include { runMACE; calcREF; updateDataset; reTrainMACE } from './processes.nf'

workflow ITERATION_STEP {
    take:
        input_ch

    main:
	// Input scripts and files
	propagator      = file('scripts/MTD_committee_plumed_MACE_system.py')
	descriptor      = file('scripts/MACE_compare_descriptors.py')
	prepare         = file('scripts/prepare_cp2k_farming_jobs.py')
	parse           = file('scripts/parse_cp2k_farmed_to_extxyz.py')
	cp2k_template   = file('input/template.inp')
	init_frame      = file('input/TDMAS_SiO2_start.traj')
	f_model         = Channel.value(file('input/MACE_models/mace-mpa-0-medium.model'))
	seed_channel    = Channel.from(123, 456, 789)

	// Initial models
	initial_models = Channel.value([
	  '/scratch/project_462000838/active_learning_nextflow/input/MACE_models/MACE_model_03_finetuned.model',
	  '/scratch/project_462000838/active_learning_nextflow/input/MACE_models/MACE_model_04_finetuned.model',
	  '/scratch/project_462000838/active_learning_nextflow/input/MACE_models/MACE_model_05_finetuned.model'
	])

	// Dataset
	persistent_dataset_path = '/scratch/project_462000838/active_learning_nextflow/growing_dataset/growing_retrain_dataset.xyz'
	evolving_dataset        = file(persistent_dataset_path)

	// === First MACE run (initial models) ===
	first_mace_out = runMACE(
	  propagator,
	  descriptor,
	  init_frame,
	  evolving_dataset,
	  initial_models,
	  'initial'
	)

	// === DFT Refinement ===
	calcREF_out = calcREF(
	  prepare,
	  parse,
	  cp2k_template,
	  first_mace_out.mace_frames
	)

	// === Update dataset ===
	update_out = updateDataset(
	  calcREF_out.new_data,
	  evolving_dataset
	)

	updated_dataset = update_out.updated_dataset

	// === Retrain MACE on new data ===
	retrain_out = reTrainMACE(
	  updated_dataset,
	  f_model,
	  seed_channel
	)

	// === Collect retrained models and filter out compiled files ===
	retrained_models = retrain_out.trained_models
	    .flatten()
	    .filter { !it.getFileName().toString().endsWith('_compiled.model') }
	    .toList()

    emit:
        new_models = retrained_models
        grown_dataset = update_out.updated_dataset  
}

